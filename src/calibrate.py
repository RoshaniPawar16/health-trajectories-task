"""Post-hoc temperature scaling of the committed transformer.

The method was fixed before running:

1. Fit one temperature T on the dev split only (split_train_dev, frac 0.1, seed 0),
   with the prediction protocol of src/evaluate.py.  Val is never used for fitting.
2. Work from the model's probabilities: p_T is proportional to p ** (1 / T),
   renormalised.  This equals softmax(logits / T); masked tokens stay at zero.
3. Choose T by minimising mean dev NLL over a grid from 0.50 to 3.00 in steps of 0.01.
4. Evaluate on val with evaluate() under the name transformer_tempscaled.
5. Calibration on val with the decade bins of outputs/calibration.csv, before and
   after, plus per-disease observed/expected (O/E), before and after.

Success criterion, declared before running: the result counts as a fix only if the
top bin's observed/predicted ratio moves closer to 1 on val.

Correctness check, before anything is written: with T = 1 the wrapped predict
function must reproduce the committed transformer mean NLL within 1e-6 and the
committed per-disease AUROCs within 1e-9.

Prerequisites: checkpoints/best.pt and the outputs of src.run_eval and src.analysis.

Writes to outputs/:
    transformer_tempscaled_per_disease_auc.csv, transformer_tempscaled_stratified_auc.csv  via evaluate()
    calibration_tempscaled.csv   one row per probability bin, before and after
    calibration_tempscaled.json  T, dev NLLs, val metrics, O/E summaries, top-bin verdict
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch

from src.bootstrap import all_aurocs, group_scores
from src.data import build_sequences, load_split, load_vocab, split_train_dev
from src.evaluate import evaluate
from src.model import HealthTransformer, make_predict_fn, select_device

OUTPUTS_DIR = Path("outputs")
CHECKPOINT_PATH = Path("checkpoints/best.pt")
_EXCLUDED_TRAIN_PATIENT: int = 402867
_BATCH: int = 256             # patients per batch, same as evaluate.py
_P_FLOOR: float = 1e-12       # probability floor for log(), same clip as evaluate.py
_T_GRID: np.ndarray = np.round(np.arange(50, 301) / 100.0, 2)  # 0.50, 0.51, ..., 3.00
_NLL_TOL: float = 1e-6        # T = 1 mean NLL must match model_comparison.csv within this
_AUC_TOL: float = 1e-9        # T = 1 per-disease AUROC must match the committed CSV below this
_PATH_TOL: float = 1e-6       # general path at T = 1 must match the untouched probabilities within this

Sequence = tuple[np.ndarray, np.ndarray]
PredictFn = Callable[[list[Sequence]], np.ndarray]


# ---------------------------------------------------------------------------
# Temperature scaling of probabilities
# ---------------------------------------------------------------------------

def scale_probs(probs: np.ndarray, T: float) -> np.ndarray:
    """Return probs ** (1 / T), renormalised per row; equal to softmax(logits / T).

    Computed in float64, returned as float32.  A probability of exactly zero
    (padding and sex tokens) stays zero.
    """
    p = probs.astype(np.float64) ** (1.0 / T)
    p /= p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def make_scaled_fn(base_fn: PredictFn, T: float) -> PredictFn:
    """Wrap a predict_fn with temperature T.

    At T exactly 1 the base probabilities are returned untouched, so the wrapper
    reproduces the committed results bit for bit.  Renormalising at T = 1 would
    divide each row by a sum that is not exactly 1 in float32 and flip near-ties
    between rows, which moves per-disease AUROC by more than the check tolerance.
    """
    def predict(prefixes: list[Sequence]) -> np.ndarray:
        probs = base_fn(prefixes)
        if T == 1.0:
            return probs
        return scale_probs(probs, T)

    return predict


# ---------------------------------------------------------------------------
# Prediction points, same protocol as evaluate.py
# ---------------------------------------------------------------------------

def prediction_batches(sequences: list[Sequence], batch_size: int = _BATCH) -> Iterator[tuple[list[Sequence], np.ndarray]]:
    """Yield (prefixes, true tokens) per batch of patients; at point k the prefix is positions 0..k-1."""
    for batch_start in range(0, len(sequences), batch_size):
        prefixes: list[Sequence] = []
        true_toks: list[int] = []
        for tokens, ages in sequences[batch_start : batch_start + batch_size]:
            if len(tokens) < 2:
                continue
            for k in range(1, len(tokens)):
                prefixes.append((tokens[:k], ages[:k]))
                true_toks.append(int(tokens[k]))
        if prefixes:
            yield prefixes, np.array(true_toks, dtype=np.int64)


def n_prediction_points(sequences: list[Sequence]) -> int:
    """Number of prediction points: L - 1 per patient with at least two events."""
    return sum(len(tokens) - 1 for tokens, _ in sequences if len(tokens) >= 2)


# ---------------------------------------------------------------------------
# Fit T on dev
# ---------------------------------------------------------------------------

def collect_log_probs(base_fn: PredictFn, sequences: list[Sequence], vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Log of the base probabilities for every prediction point of sequences.

    Returns:
        logp     (n_points, vocab_size) float64; minus infinity where the probability is zero
        true_tok (n_points,) int64
    """
    n_points = n_prediction_points(sequences)
    logp = np.empty((n_points, vocab_size), dtype=np.float64)
    true_tok = np.empty(n_points, dtype=np.int64)
    filled = 0
    for prefixes, true_arr in prediction_batches(sequences):
        probs = base_fn(prefixes).astype(np.float64)
        n = len(true_arr)
        with np.errstate(divide="ignore"):  # log(0) = -inf is intended for masked tokens
            logp[filled : filled + n] = np.log(probs)
        true_tok[filled : filled + n] = true_arr
        filled += n
    return logp, true_tok


def mean_nll_at(logp: np.ndarray, true_tok: np.ndarray, T: float) -> float:
    """Mean NLL after temperature T, from stored log probabilities.

    log p_T = logp / T - logsumexp(logp / T), which is the log of p ** (1 / T) renormalised.
    The true-token probability is clipped at _P_FLOOR, as evaluate.py does.
    """
    scaled = logp / T
    row_max = scaled.max(axis=1, keepdims=True)
    log_norm = row_max[:, 0] + np.log(np.exp(scaled - row_max).sum(axis=1))
    logp_true = scaled[np.arange(len(true_tok)), true_tok] - log_norm
    p_true = np.clip(np.exp(logp_true), _P_FLOOR, None)
    return float(-np.log(p_true).mean())


def fit_temperature(logp: np.ndarray, true_tok: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Grid search for the T that minimises mean NLL.

    Returns:
        (best T, DataFrame with columns T and dev_nll for the whole grid).
        Ties go to the smallest T.
    """
    rows = [{"T": float(T), "dev_nll": mean_nll_at(logp, true_tok, float(T))} for T in _T_GRID]
    grid = pd.DataFrame(rows)
    best_T = float(grid.loc[grid["dev_nll"].idxmin(), "T"])
    return best_T, grid


# ---------------------------------------------------------------------------
# Val pass and calibration tables
# ---------------------------------------------------------------------------

def collect_val(
    base_fn: PredictFn,
    sequences: list[Sequence],
    eligible_arr: np.ndarray,
    T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """One pass over val: base and scaled probabilities of the eligible diseases.

    Returns:
        nll_base     (n_points,) float64, per-point NLL of the base model (T = 1)
        true_tok     (n_points,) int64
        probs_base   (n_points, n_eligible) float32, untouched base probabilities
        probs_scaled (n_points, n_eligible) float32, after temperature T
        path_diff    largest |scale_probs(p, 1) - p| over all entries: how far the
                     general scaling path at T = 1 is from the untouched probabilities
    """
    n_points = n_prediction_points(sequences)
    nll_base = np.empty(n_points, dtype=np.float64)
    true_tok = np.empty(n_points, dtype=np.int64)
    probs_base = np.empty((n_points, len(eligible_arr)), dtype=np.float32)
    probs_scaled = np.empty((n_points, len(eligible_arr)), dtype=np.float32)
    scaled_fn = make_scaled_fn(lambda x: x, T)  # applied to probabilities already computed
    path_diff = 0.0
    filled = 0
    for prefixes, true_arr in prediction_batches(sequences):
        probs = base_fn(prefixes)
        n = len(true_arr)
        true_probs = np.clip(probs[np.arange(n), true_arr], _P_FLOOR, None)
        nll_base[filled : filled + n] = -np.log(true_probs)
        true_tok[filled : filled + n] = true_arr
        probs_base[filled : filled + n] = probs[:, eligible_arr]
        probs_scaled[filled : filled + n] = scaled_fn(probs)[:, eligible_arr]
        path_diff = max(path_diff, float(np.abs(scale_probs(probs, 1.0) - probs).max()))
        filled += n
    return nll_base, true_tok, probs_base, probs_scaled, path_diff


def calibration_table(
    probs: np.ndarray,
    true_tok: np.ndarray,
    eligible_list: list[int],
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin every (prediction point, eligible disease) pair by predicted probability.

    Same binning as src/analysis.py: np.digitize against the inner edges, so values
    below the first edge fall in the first bin and values at or above the last inner
    edge fall in the last bin.

    Returns:
        mean_pred (n_bins,) mean predicted probability per bin, NaN for an empty bin
        obs_rate  (n_bins,) observed rate per bin, NaN for an empty bin
        counts    (n_bins,) number of pairs per bin
        o_e       (n_eligible,) per-disease observed / expected = positives / sum of predicted
    """
    n_bins = len(edges) - 1
    sum_pred = np.zeros(n_bins, dtype=np.float64)
    sum_obs = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    o_e = np.empty(len(eligible_list), dtype=np.float64)
    for d, tok in enumerate(eligible_list):
        d_probs = probs[:, d].astype(np.float64)
        d_labels = (true_tok == tok).astype(np.float64)
        bids = np.digitize(d_probs, edges[1:-1])
        sum_pred += np.bincount(bids, weights=d_probs, minlength=n_bins)
        sum_obs += np.bincount(bids, weights=d_labels, minlength=n_bins)
        counts += np.bincount(bids, minlength=n_bins)
        o_e[d] = d_labels.sum() / d_probs.sum()
    nonzero = counts > 0
    mean_pred = np.where(nonzero, sum_pred / np.where(nonzero, counts, 1.0), np.nan)
    obs_rate = np.where(nonzero, sum_obs / np.where(nonzero, counts, 1.0), np.nan)
    return mean_pred, obs_rate, counts.astype(np.int64), o_e


def o_e_summary(o_e: np.ndarray) -> dict:
    """Median and quartiles of the per-disease O/E ratios."""
    q25, med, q75 = (float(q) for q in np.percentile(o_e, [25, 50, 75]))
    return {"median": med, "q25": q25, "q75": q75}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_df = load_split("train")
    val_df = load_split("val")
    vocab = load_vocab()
    train_df = train_df[train_df["person_id"] != _EXCLUDED_TRAIN_PATIENT]
    train_sequences = build_sequences(train_df)
    val_sequences = build_sequences(val_df)
    unseen: set[int] = set(val_df["token"]) - set(train_df["token"])  # same exclusion as run_eval.py

    # Dev split: the same patients that early stopping and K tuning used
    _, dev_part = split_train_dev(train_sequences)
    print(f"dev split: {len(dev_part):,} patients, {n_prediction_points(dev_part):,} prediction points")

    # Committed transformer, loaded as run_eval.py loads it
    device = select_device()
    print(f"Device: {device}")
    model = HealthTransformer()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)
    base_fn = make_predict_fn(model, device, vocab)

    # --- Fit T on dev only ---
    dev_logp, dev_true = collect_log_probs(base_fn, dev_part, len(vocab))
    best_T, grid = fit_temperature(dev_logp, dev_true)
    dev_nll_1 = float(grid.loc[grid["T"] == 1.0, "dev_nll"].iloc[0])
    dev_nll_T = float(grid.loc[grid["T"] == best_T, "dev_nll"].iloc[0])
    on_edge = bool(best_T in (float(_T_GRID[0]), float(_T_GRID[-1])))
    del dev_logp
    print(f"chosen T = {best_T:.2f}  (grid {_T_GRID[0]:.2f} to {_T_GRID[-1]:.2f}, step 0.01)")
    print(f"dev NLL at T = 1: {dev_nll_1:.6f}   dev NLL at T = {best_T:.2f}: {dev_nll_T:.6f}")
    if on_edge:
        print("WARNING: the chosen T is on the edge of the grid; the true minimum may lie outside it")

    # --- One pass over val: base and scaled probabilities of the eligible diseases ---
    elig_csv = pd.read_csv(OUTPUTS_DIR / "transformer_per_disease_auc.csv")
    elig_names = set(elig_csv["disease"])
    eligible_list = [i for i, v in enumerate(vocab) if v in elig_names]
    if len(eligible_list) != len(elig_names):
        raise RuntimeError("eligible disease names do not map one-to-one onto the vocabulary")
    eligible_arr = np.array(eligible_list, dtype=np.int64)
    nll_base, true_tok, probs_base, probs_scaled, path_diff = collect_val(base_fn, val_sequences, eligible_arr, best_T)

    # --- Correctness check at T = 1, before anything is written ---
    mc = pd.read_csv(OUTPUTS_DIR / "model_comparison.csv").set_index("name")
    committed = mc.loc["transformer"]
    d_nll = abs(float(nll_base.mean()) - float(committed["mean_nll"]))

    gids, n_groups = group_scores(probs_base)
    pos_lists = [np.flatnonzero(true_tok == tok) for tok in eligible_list]
    auc_base = all_aurocs(gids, n_groups, pos_lists, np.ones(len(true_tok), dtype=np.float64))
    del gids
    ref_auc = elig_csv.set_index("disease")["auc"].loc[[vocab[i] for i in eligible_list]].to_numpy()
    d_auc = float(np.abs(auc_base - ref_auc).max())

    print("correctness check at T = 1 (wrapper returns the base probabilities untouched):")
    print(f"  |mean NLL diff| vs model_comparison.csv: {d_nll:.3e}  (must be <= {_NLL_TOL:.0e})")
    print(f"  max |AUROC diff| vs transformer_per_disease_auc.csv: {d_auc:.3e}  (must be < {_AUC_TOL:.0e})")
    print(f"  general scaling path at T = 1, max |p ** 1 renormalised - p|: {path_diff:.3e}  (must be <= {_PATH_TOL:.0e})")
    # "not (x < tol)" also catches NaN
    if not (d_nll <= _NLL_TOL) or not (d_auc < _AUC_TOL) or not (path_diff <= _PATH_TOL):
        raise RuntimeError("calibrate correctness check failed; nothing was written")
    print("  OK")

    # --- Calibration before and after, with the bins of the committed calibration.csv ---
    cal_ref = pd.read_csv(OUTPUTS_DIR / "calibration.csv")
    edges = np.append(cal_ref["lower_edge"].to_numpy(), cal_ref["upper_edge"].iloc[-1])
    pred_b, obs_b, cnt_b, oe_b = calibration_table(probs_base, true_tok, eligible_list, edges)
    if not np.array_equal(cnt_b, cal_ref["count"].to_numpy()):
        raise RuntimeError("bin counts at T = 1 differ from outputs/calibration.csv; nothing was written")
    pred_a, obs_a, cnt_a, oe_a = calibration_table(probs_scaled, true_tok, eligible_list, edges)
    del probs_base, probs_scaled

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_b = obs_b / pred_b   # observed / predicted; below 1 = overconfident in that bin
        ratio_a = obs_a / pred_a
    top = len(edges) - 2           # index of the highest-probability bin
    closer = bool(abs(ratio_a[top] - 1.0) < abs(ratio_b[top] - 1.0))

    # --- Evaluate on val under the declared name (writes the two transformer_tempscaled CSVs) ---
    result = evaluate(
        make_scaled_fn(base_fn, best_T), val_sequences, vocab, "transformer_tempscaled",
        excluded_from_auroc=unseen,
    )

    # --- Write ---
    cal_out = pd.DataFrame({
        "bin": np.arange(len(edges) - 1),
        "lower_edge": edges[:-1],
        "upper_edge": edges[1:],
        "mean_predicted_before": pred_b,
        "observed_rate_before": obs_b,
        "count_before": cnt_b,
        "observed_over_predicted_before": ratio_b,
        "mean_predicted_after": pred_a,
        "observed_rate_after": obs_a,
        "count_after": cnt_a,
        "observed_over_predicted_after": ratio_a,
    })
    cal_out.to_csv(OUTPUTS_DIR / "calibration_tempscaled.csv", index=False)

    metric_keys = ("mean_nll", "top1_acc", "top5_acc", "top20_acc", "mean_auroc", "median_auroc")
    summary = {
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(device),
        "temperature": best_T,
        "grid": {"min": float(_T_GRID[0]), "max": float(_T_GRID[-1]), "step": 0.01},
        "temperature_on_grid_edge": on_edge,
        "fit_split": "dev (split_train_dev, frac 0.1, seed 0); val is not used for fitting",
        "dev": {
            "n_patients": int(len(dev_part)),
            "n_points": int(len(dev_true)),
            "nll_at_T_1": dev_nll_1,
            "nll_at_T": dev_nll_T,
        },
        "correctness_check_at_T_1": {
            "nll_diff": d_nll,
            "max_auc_diff": d_auc,
            "general_path_max_prob_diff": path_diff,
        },
        "val_before": {k: float(committed[k]) for k in metric_keys},   # from model_comparison.csv
        "val_after": {k: float(result[k]) for k in metric_keys},       # from evaluate()
        "n_diseases_auroc_after": int(result["n_diseases_auroc"]),
        "o_e_before": o_e_summary(oe_b),
        "o_e_after": o_e_summary(oe_a),
        "top_bin": {
            "lower_edge": float(edges[top]),
            "upper_edge": float(edges[top + 1]),
            "before": {
                "mean_predicted": float(pred_b[top]), "observed_rate": float(obs_b[top]),
                "count": int(cnt_b[top]), "observed_over_predicted": float(ratio_b[top]),
            },
            "after": {
                "mean_predicted": float(pred_a[top]), "observed_rate": float(obs_a[top]),
                "count": int(cnt_a[top]), "observed_over_predicted": float(ratio_a[top]),
            },
        },
        "success_criterion": "top bin observed/predicted ratio moves closer to 1 on val",
        "top_bin_ratio_closer_to_1": closer,
    }
    with open(OUTPUTS_DIR / "calibration_tempscaled.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Report ---
    print("\n=== Temperature scaling on val ===")
    print(f"T = {best_T:.2f}, fitted on dev")
    for k in metric_keys:
        print(f"{k:<14} before {float(committed[k]):.4f}   after {float(result[k]):.4f}")
    print(f"per-disease O/E before: median {summary['o_e_before']['median']:.4f}, IQR [{summary['o_e_before']['q25']:.4f}, {summary['o_e_before']['q75']:.4f}]")
    print(f"per-disease O/E after:  median {summary['o_e_after']['median']:.4f}, IQR [{summary['o_e_after']['q25']:.4f}, {summary['o_e_after']['q75']:.4f}]")
    print(
        f"top bin [{edges[top]:.0e}, {edges[top + 1]:.0e}): observed/predicted before {ratio_b[top]:.3f} (n = {int(cnt_b[top]):,}),"
        f" after {ratio_a[top]:.3f} (n = {int(cnt_a[top]):,})"
    )
    print(f"declared criterion, top bin ratio closer to 1: {'met' if closer else 'not met'}")
    print(f"Saved {OUTPUTS_DIR / 'calibration_tempscaled.csv'}")
    print(f"Saved {OUTPUTS_DIR / 'calibration_tempscaled.json'}")
