"""Bootstrap confidence intervals over validation patients, transformer vs age_sex.

Prerequisites: checkpoints/best.pt and the outputs of python -m src.baselines,
python -m src.run_eval and python -m src.analysis must exist.

The prediction protocol is the one in src/evaluate.py: at point k the model sees
positions 0..k-1.  Probabilities are computed once.  Each resample of val patients
becomes a weight per prediction point (how many times its patient was drawn); no
rows are copied.

Writes to outputs/ (only if the correctness check passes):
    bootstrap_summary.json  point estimates, 95% intervals, n_resamples, seed, skip counts
    bootstrap_panel.csv     cardiometabolic panel, one row per code
    bootstrap_draws.csv     one row per resample
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.baselines import fit_age_sex
from src.data import build_sequences, load_split, load_vocab
from src.model import HealthTransformer, make_predict_fn, select_device

OUTPUTS_DIR = Path("outputs")
CHECKPOINT_PATH = Path("checkpoints/best.pt")
_EXCLUDED_TRAIN_PATIENT: int = 402867
_BATCH: int = 256            # patients per batch, same as evaluate.py
_P_FLOOR: float = 1e-12      # probability floor for log(), same clip as evaluate.py
_SEED: int = 0               # seed for the patient resampling
_AUC_TOL: float = 1e-9       # unit-weight AUROC must match the committed CSVs below this
_NLL_TOL: float = 1e-6       # unit-weight mean NLL must match model_comparison.csv within this
_N_TIMING: int = 5           # resamples timed before choosing the total
_BUDGET_SEC: float = 30 * 60.0
_N_MAX: int = 1000
_N_MIN: int = 500
_PROGRESS_EVERY: int = 50

# Cardiometabolic panel, declared before running: I20 to I25, I48, I50, I60 to I69, E11.
PANEL_CODES: list[str] = (
    [f"I{n}" for n in range(20, 26)]
    + ["I48", "I50"]
    + [f"I{n}" for n in range(60, 70)]
    + ["E11"]
)


def _log(msg: str) -> None:
    """Print immediately, so progress is visible while the script runs."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Per-point quantities, computed once per model
# ---------------------------------------------------------------------------

def collect_points(
    predict_fn: Callable[[list[tuple[np.ndarray, np.ndarray]]], np.ndarray],
    sequences: list[tuple[np.ndarray, np.ndarray]],
    eligible_arr: np.ndarray,
    batch_size: int = _BATCH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run predict_fn over every prediction point, with evaluate.py's protocol and batching.

    Args:
        predict_fn:   takes a list of (tokens, ages) prefixes, returns (n, vocab_size) probabilities.
        sequences:    output of build_sequences(val_df).
        eligible_arr: vocab indices of the AUROC-eligible diseases.
        batch_size:   patients per batch.

    Returns:
        nll         (n_points,) float64, -log of the clipped probability of the true token
        patient_idx (n_points,) int32, index into sequences of the point's patient
        true_tok    (n_points,) int64, the true next token
        probs       (n_points, n_eligible) float32, probabilities of the eligible diseases only
    """
    n_points = sum(len(tokens) - 1 for tokens, _ in sequences if len(tokens) >= 2)
    nll = np.empty(n_points, dtype=np.float64)
    patient_idx = np.empty(n_points, dtype=np.int32)
    true_tok = np.empty(n_points, dtype=np.int64)
    probs_elig = np.empty((n_points, len(eligible_arr)), dtype=np.float32)

    filled = 0
    for batch_start in range(0, len(sequences), batch_size):
        batch = sequences[batch_start : batch_start + batch_size]
        prefixes: list[tuple[np.ndarray, np.ndarray]] = []
        true_toks: list[int] = []
        pat: list[int] = []
        for offset, (tokens, ages) in enumerate(batch):
            if len(tokens) < 2:
                continue
            for k in range(1, len(tokens)):
                prefixes.append((tokens[:k], ages[:k]))  # positions 0..k-1 only
                true_toks.append(int(tokens[k]))
                pat.append(batch_start + offset)
        if not prefixes:
            continue

        probs = predict_fn(prefixes)  # (n, vocab_size)
        n = len(prefixes)
        true_arr = np.array(true_toks, dtype=np.int64)
        true_probs = np.clip(probs[np.arange(n), true_arr], _P_FLOOR, None)

        nll[filled : filled + n] = -np.log(true_probs)
        patient_idx[filled : filled + n] = pat
        true_tok[filled : filled + n] = true_arr
        # float32, exactly as evaluate.py stores scores before AUROC, so ties are identical
        probs_elig[filled : filled + n] = probs[:, eligible_arr].astype(np.float32)
        filled += n

    if filled != n_points:
        raise RuntimeError(f"collect_points: filled {filled} of {n_points} points")
    return nll, patient_idx, true_tok, probs_elig


def group_scores(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort each disease's scores once and give tied scores the same group id.

    Args:
        probs: (n_points, n_eligible) float32 scores.

    Returns:
        gids     (n_eligible, n_points) int32; group id of each point, ascending in score
        n_groups (n_eligible,) int64; number of distinct scores per disease
    """
    n_points, n_dis = probs.shape
    gids = np.empty((n_dis, n_points), dtype=np.int32)
    n_groups = np.empty(n_dis, dtype=np.int64)
    for d in range(n_dis):
        uniq, inverse = np.unique(probs[:, d], return_inverse=True)
        gids[d] = inverse.reshape(-1)
        n_groups[d] = len(uniq)
    return gids, n_groups


# ---------------------------------------------------------------------------
# Weighted AUROC
# ---------------------------------------------------------------------------

def weighted_auroc(gid: np.ndarray, n_group: int, pos_idx: np.ndarray, w: np.ndarray) -> float:
    """Weighted AUROC for one disease, with tied scores counted as half.

    Args:
        gid:     (n_points,) group id of each point; equal scores share an id, ids ascend with score.
        n_group: number of distinct scores.
        pos_idx: indices of the points whose true next token is this disease.
        w:       (n_points,) weight of each point (times its patient was drawn).

    Returns:
        AUROC, or NaN if the positive or the negative weight is zero.
    """
    total = np.bincount(gid, weights=w, minlength=n_group)
    pos = np.bincount(gid[pos_idx], weights=w[pos_idx], minlength=n_group)
    neg = total - pos
    pos_total = pos.sum()
    neg_total = neg.sum()
    if pos_total == 0 or neg_total == 0:
        return float("nan")
    neg_below = np.cumsum(neg) - neg
    numerator = (pos * (neg_below + 0.5 * neg)).sum()
    return float(numerator / (pos_total * neg_total))


def all_aurocs(
    gids: np.ndarray,
    n_groups: np.ndarray,
    pos_lists: list[np.ndarray],
    w: np.ndarray,
) -> np.ndarray:
    """Weighted AUROC for every eligible disease; NaN where a disease is skipped."""
    out = np.empty(len(pos_lists), dtype=np.float64)
    for d, pos_idx in enumerate(pos_lists):
        out[d] = weighted_auroc(gids[d], int(n_groups[d]), pos_idx, w)
    return out


# ---------------------------------------------------------------------------
# Correctness check, resampling helpers
# ---------------------------------------------------------------------------

def check_against_committed(
    eligible_names: list[str],
    auc_t: np.ndarray,
    auc_a: np.ndarray,
    nll_t: float,
    nll_a: float,
) -> dict:
    """Compare unit-weight results with the committed files; raise if they disagree.

    Args:
        eligible_names: disease names in the order of auc_t and auc_a.
        auc_t, auc_a:   unit-weight per-disease AUROC for transformer and age_sex.
        nll_t, nll_a:   unit-weight mean NLL for transformer and age_sex.

    Returns:
        Dict with the three maximum absolute differences.
    """
    ref_t = pd.read_csv(OUTPUTS_DIR / "transformer_per_disease_auc.csv").set_index("disease")["auc"]
    ref_a = pd.read_csv(OUTPUTS_DIR / "age_sex_per_disease_auc.csv").set_index("disease")["auc"]
    mc = pd.read_csv(OUTPUTS_DIR / "model_comparison.csv").set_index("name")["mean_nll"]

    d_auc_t = float(np.abs(auc_t - ref_t.loc[eligible_names].to_numpy()).max())
    d_auc_a = float(np.abs(auc_a - ref_a.loc[eligible_names].to_numpy()).max())
    d_nll_t = abs(nll_t - float(mc.loc["transformer"]))
    d_nll_a = abs(nll_a - float(mc.loc["age_sex"]))
    max_auc = max(d_auc_t, d_auc_a)
    max_nll = max(d_nll_t, d_nll_a)

    _log("correctness check, all weights equal to 1:")
    _log(f"  max |AUROC diff| vs committed CSVs: {max_auc:.3e}  (transformer {d_auc_t:.3e}, age_sex {d_auc_a:.3e}; must be < {_AUC_TOL:.0e})")
    _log(f"  max |mean NLL diff| vs model_comparison.csv: {max_nll:.3e}  (transformer {d_nll_t:.3e}, age_sex {d_nll_a:.3e}; must be <= {_NLL_TOL:.0e})")
    # "not (x < tol)" also catches NaN
    if not (max_auc < _AUC_TOL) or not (max_nll <= _NLL_TOL):
        raise RuntimeError("bootstrap correctness check failed; nothing was written")
    _log("  OK")
    return {
        "max_auc_diff_transformer": d_auc_t,
        "max_auc_diff_age_sex": d_auc_a,
        "max_nll_diff_transformer": d_nll_t,
        "max_nll_diff_age_sex": d_nll_a,
    }


def draw_weights(rng: np.random.Generator, n_patients: int, patient_idx: np.ndarray) -> np.ndarray:
    """One resample of patients with replacement, returned as a weight per prediction point."""
    drawn = rng.integers(0, n_patients, size=n_patients)
    counts = np.bincount(drawn, minlength=n_patients)
    return counts[patient_idx].astype(np.float64)


def choose_n_resamples(sec_per_resample: float, setup_sec: float) -> tuple[int, str]:
    """Pick the number of resamples from the measured time per resample.

    1000 if setup plus 1000 resamples is projected under 30 minutes; otherwise the
    largest multiple of 100 that fits, with a minimum of 500.
    """
    projected = setup_sec + sec_per_resample * _N_MAX
    if projected < _BUDGET_SEC:
        return _N_MAX, (
            f"{_N_MAX} resamples: projected total {projected / 60:.1f} min is under the "
            f"{_BUDGET_SEC / 60:.0f} min budget"
        )
    fits = int((_BUDGET_SEC - setup_sec) // sec_per_resample // 100) * 100
    n = max(_N_MIN, min(fits, _N_MAX))
    return n, (
        f"{n} resamples: {_N_MAX} would take {projected / 60:.1f} min, over the "
        f"{_BUDGET_SEC / 60:.0f} min budget; largest multiple of 100 that fits is {max(fits, 0)}, minimum {_N_MIN}"
    )


def percentile_interval(x: np.ndarray) -> tuple[float, float]:
    """95 percent percentile interval, ignoring NaN (skipped resamples)."""
    if np.isnan(x).all():
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def build_panel(
    vocab: list[str],
    eligible_list: list[int],
    true_tok: np.ndarray,
    point_t: np.ndarray,
    point_a: np.ndarray,
    draws_t: np.ndarray,
    draws_a: np.ndarray,
) -> pd.DataFrame:
    """Cardiometabolic panel: one row per vocabulary entry matching a panel code.

    A code that is not eligible, or not in the vocabulary at all, still gets a row
    with status "not eligible".

    Args:
        vocab:          output of load_vocab().
        eligible_list:  vocab indices of the eligible diseases, in column order.
        true_tok:       (n_points,) true next token, to count positives of ineligible codes.
        point_t/a:      (n_eligible,) unit-weight AUROC per disease.
        draws_t/a:      (n_resamples, n_eligible) AUROC per resample, NaN where skipped.
    """
    col_of = {tok: d for d, tok in enumerate(eligible_list)}
    rows: list[dict] = []
    for code in PANEL_CODES:
        matches = [i for i, v in enumerate(vocab) if v[:3] == code]
        if not matches:
            rows.append({"code": code, "disease": "", "status": "not eligible", "note": "code not in vocabulary"})
            continue
        for tok in matches:
            n_pos = int((true_tok == tok).sum())
            if tok not in col_of:
                rows.append({
                    "code": code, "disease": vocab[tok], "status": "not eligible",
                    "note": "not in outputs/transformer_per_disease_auc.csv", "n_positives": n_pos,
                })
                continue
            d = col_of[tok]
            diff_draws = draws_t[:, d] - draws_a[:, d]
            t_lo, t_hi = percentile_interval(draws_t[:, d])
            a_lo, a_hi = percentile_interval(draws_a[:, d])
            d_lo, d_hi = percentile_interval(diff_draws)
            rows.append({
                "code": code, "disease": vocab[tok], "status": "eligible", "note": "",
                "n_positives": n_pos,
                "auc_transformer": float(point_t[d]), "auc_transformer_lo": t_lo, "auc_transformer_hi": t_hi,
                "auc_age_sex": float(point_a[d]), "auc_age_sex_lo": a_lo, "auc_age_sex_hi": a_hi,
                "diff": float(point_t[d] - point_a[d]), "diff_lo": d_lo, "diff_hi": d_hi,
                "n_resamples_used": int((~np.isnan(diff_draws)).sum()),
            })
    columns = [
        "code", "disease", "status", "note", "n_positives",
        "auc_transformer", "auc_transformer_lo", "auc_transformer_hi",
        "auc_age_sex", "auc_age_sex_lo", "auc_age_sex_hi",
        "diff", "diff_lo", "diff_hi", "n_resamples_used",
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute per-point quantities, check them, bootstrap, write the three outputs."""
    t_start = time.time()

    # --- Load ---
    t0 = time.time()
    train_df = load_split("train")
    val_df = load_split("val")
    vocab = load_vocab()
    train_df = train_df[train_df["person_id"] != _EXCLUDED_TRAIN_PATIENT]
    train_sequences = build_sequences(train_df)
    val_sequences = build_sequences(val_df)
    sex_indices = {i for i, v in enumerate(vocab) if v in ("Female", "Male")}
    n_patients = len(val_sequences)

    # Eligible diseases: the rows of transformer_per_disease_auc.csv, in vocab index order
    elig_names_csv = set(pd.read_csv(OUTPUTS_DIR / "transformer_per_disease_auc.csv")["disease"])
    elig_names_age_sex = set(pd.read_csv(OUTPUTS_DIR / "age_sex_per_disease_auc.csv")["disease"])
    if elig_names_csv != elig_names_age_sex:
        raise RuntimeError("transformer and age_sex per-disease CSVs list different diseases")
    eligible_list = [i for i, v in enumerate(vocab) if v in elig_names_csv]
    if len(eligible_list) != len(elig_names_csv):
        raise RuntimeError("eligible disease names do not map one-to-one onto the vocabulary")
    eligible_arr = np.array(eligible_list, dtype=np.int64)
    eligible_names = [vocab[i] for i in eligible_list]
    n_elig = len(eligible_list)
    _log(f"[load] {n_patients:,} val patients, {n_elig} eligible diseases  ({time.time() - t0:.1f}s)")

    # --- age_sex, refit with the K recorded by analysis.py ---
    t0 = time.time()
    summary_in = json.loads((OUTPUTS_DIR / "analysis_summary.json").read_text())
    best_K = float(summary_in["provenance"]["age_sex_K"])
    age_sex_fn = fit_age_sex(train_sequences, vocab, sex_indices, K=best_K)
    _log(f"[fit] age_sex refit on train minus patient {_EXCLUDED_TRAIN_PATIENT}, K={best_K:g}  ({time.time() - t0:.1f}s)")

    # --- Transformer, same loading and device selection as run_eval.py ---
    device = select_device()
    model = HealthTransformer()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)
    transformer_fn = make_predict_fn(model, device, vocab)
    _log(f"[model] transformer loaded from {CHECKPOINT_PATH} on {device}")

    # --- Per-point quantities; probabilities are dropped once grouped ---
    t0 = time.time()
    nll_t, patient_idx, true_tok, probs = collect_points(transformer_fn, val_sequences, eligible_arr)
    _log(f"[points] transformer: {len(nll_t):,} prediction points  ({time.time() - t0:.1f}s)")
    t0 = time.time()
    gids_t, n_groups_t = group_scores(probs)
    del probs
    _log(f"[group] transformer scores sorted and tie-grouped  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    nll_a, patient_idx_a, true_tok_a, probs = collect_points(age_sex_fn, val_sequences, eligible_arr)
    if not (np.array_equal(patient_idx, patient_idx_a) and np.array_equal(true_tok, true_tok_a)):
        raise RuntimeError("the two models were scored on different prediction points")
    _log(f"[points] age_sex: {len(nll_a):,} prediction points  ({time.time() - t0:.1f}s)")
    t0 = time.time()
    gids_a, n_groups_a = group_scores(probs)
    del probs
    _log(f"[group] age_sex scores sorted and tie-grouped  ({time.time() - t0:.1f}s)")

    n_points = len(true_tok)
    pos_lists = [np.flatnonzero(true_tok == tok) for tok in eligible_list]

    # --- Correctness check with every weight equal to 1; raises before anything is written ---
    ones = np.ones(n_points, dtype=np.float64)
    point_auc_t = all_aurocs(gids_t, n_groups_t, pos_lists, ones)
    point_auc_a = all_aurocs(gids_a, n_groups_a, pos_lists, ones)
    point_nll_t = float(nll_t.mean())
    point_nll_a = float(nll_a.mean())
    check = check_against_committed(eligible_names, point_auc_t, point_auc_a, point_nll_t, point_nll_a)
    setup_sec = time.time() - t_start

    # --- Resampling ---
    rng = np.random.default_rng(_SEED)
    draws_auc_t = np.full((_N_MAX, n_elig), np.nan, dtype=np.float64)
    draws_auc_a = np.full((_N_MAX, n_elig), np.nan, dtype=np.float64)
    draws_nll_t = np.empty(_N_MAX, dtype=np.float64)
    draws_nll_a = np.empty(_N_MAX, dtype=np.float64)

    n_resamples = _N_MAX
    reason = ""
    sec_per_resample = float("nan")
    t_loop = time.time()
    b = 0
    while b < n_resamples:
        w = draw_weights(rng, n_patients, patient_idx)
        w_sum = w.sum()
        draws_nll_t[b] = float(w @ nll_t) / w_sum
        draws_nll_a[b] = float(w @ nll_a) / w_sum
        draws_auc_t[b] = all_aurocs(gids_t, n_groups_t, pos_lists, w)
        draws_auc_a[b] = all_aurocs(gids_a, n_groups_a, pos_lists, w)
        b += 1

        if b == _N_TIMING:
            # The timed resamples are kept as resamples 0..4, so results do not depend on timing
            sec_per_resample = (time.time() - t_loop) / _N_TIMING
            n_resamples, reason = choose_n_resamples(sec_per_resample, setup_sec)
            _log(f"[timing] setup {setup_sec:.1f}s, {sec_per_resample:.2f}s per resample over {_N_TIMING} resamples")
            _log(f"[timing] choice: {reason}")
        elif b % _PROGRESS_EVERY == 0 or b == n_resamples:
            elapsed = time.time() - t_loop
            eta = elapsed / b * (n_resamples - b)
            _log(f"[bootstrap] {b}/{n_resamples}  elapsed {elapsed / 60:.1f} min  remaining {eta / 60:.1f} min")

    draws_auc_t = draws_auc_t[:n_resamples]
    draws_auc_a = draws_auc_a[:n_resamples]
    draws_nll_t = draws_nll_t[:n_resamples]
    draws_nll_a = draws_nll_a[:n_resamples]

    # --- Per-resample statistics ---
    # A disease is skipped when its positive or negative weight is zero.  That depends only on
    # labels and weights, so both models skip the same diseases in a resample.
    skipped = np.isnan(draws_auc_t)
    if not np.array_equal(skipped, np.isnan(draws_auc_a)):
        raise RuntimeError("the two models skipped different diseases in some resample")
    n_used = (~skipped).sum(axis=1)
    if (n_used == 0).any():
        raise RuntimeError("a resample skipped every disease")

    draws = pd.DataFrame({
        "resample": np.arange(n_resamples),
        "nll_transformer": draws_nll_t,
        "nll_age_sex": draws_nll_a,
        "nll_diff": draws_nll_a - draws_nll_t,  # age_sex minus transformer; positive = transformer better
        "mean_auroc_transformer": np.nanmean(draws_auc_t, axis=1),
        "mean_auroc_age_sex": np.nanmean(draws_auc_a, axis=1),
        "mean_auroc_diff": np.nanmean(draws_auc_t - draws_auc_a, axis=1),
        # NaN > NaN is False, so skipped diseases never count as wins; divide by diseases used
        "frac_transformer_higher": (draws_auc_t > draws_auc_a).sum(axis=1) / n_used,
        "n_diseases_used": n_used,
    })

    point = {
        "nll_transformer": point_nll_t,
        "nll_age_sex": point_nll_a,
        "nll_diff": point_nll_a - point_nll_t,
        "mean_auroc_transformer": float(point_auc_t.mean()),
        "mean_auroc_age_sex": float(point_auc_a.mean()),
        "mean_auroc_diff": float((point_auc_t - point_auc_a).mean()),
        "frac_transformer_higher": float((point_auc_t > point_auc_a).mean()),
    }
    metrics: dict[str, dict] = {}
    for key, value in point.items():
        lo, hi = percentile_interval(draws[key].to_numpy())
        metrics[key] = {"point": float(value), "ci95_lo": lo, "ci95_hi": hi}

    skips_per_disease = skipped.sum(axis=0)
    summary = {
        "seed": int(_SEED),
        "n_resamples": int(n_resamples),
        "n_resamples_reason": reason,
        "sec_per_resample_timed": float(sec_per_resample),
        "interval": "95 percent percentile interval over resamples of val patients",
        "device": str(device),
        "age_sex_K": float(best_K),
        "n_val_patients": int(n_patients),
        "n_points": int(n_points),
        "n_eligible_diseases": int(n_elig),
        "correctness_check": check,
        "metrics": metrics,
        "skips": {
            "total_disease_resample_pairs_skipped": int(skipped.sum()),
            "total_disease_resample_pairs": int(skipped.size),
            "n_resamples_with_any_skip": int(skipped.any(axis=1).sum()),
            "per_disease": {
                eligible_names[d]: int(skips_per_disease[d]) for d in range(n_elig) if skips_per_disease[d] > 0
            },
        },
    }

    panel = build_panel(vocab, eligible_list, true_tok, point_auc_t, point_auc_a, draws_auc_t, draws_auc_a)

    # --- Write (only reached if the correctness check passed) ---
    with open(OUTPUTS_DIR / "bootstrap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    panel.to_csv(OUTPUTS_DIR / "bootstrap_panel.csv", index=False)
    draws.to_csv(OUTPUTS_DIR / "bootstrap_draws.csv", index=False)

    _log(f"\n=== Bootstrap over val patients: {n_resamples} resamples, seed {_SEED} ===")
    for key, m in metrics.items():
        _log(f"{key:<26} {m['point']:.4f}  95% CI [{m['ci95_lo']:.4f}, {m['ci95_hi']:.4f}]")
    _log(
        f"skipped disease-resample pairs: {summary['skips']['total_disease_resample_pairs_skipped']:,}"
        f" of {summary['skips']['total_disease_resample_pairs']:,}"
    )
    for name in ("bootstrap_summary.json", "bootstrap_panel.csv", "bootstrap_draws.csv"):
        _log(f"Saved {OUTPUTS_DIR / name}")
    _log(f"total runtime {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
