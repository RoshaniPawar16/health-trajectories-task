"""Seven post-training analyses comparing transformer vs age_sex baseline on val sequences.

Prerequisites: outputs from python -m src.baselines and python -m src.run_eval must exist.

Writes to outputs/:
    per_disease_paired.csv, per_disease_scatter.png                     (analysis 1)
    stratified_comparison.csv, stratified_bar.png                       (analysis 2)
    transformer_shuffled_*.csv, age_sex_shuffled_*.csv  via evaluate()  (analysis 3)
    history_length.csv, history_length.png                              (analysis 5)
    calibration.csv, calibration_per_disease.csv, calibration.png       (analysis 6)
    failures.csv, auc_vs_prevalence.png                                 (analysis 7)
    analysis_summary.json  — every number used by the printed summary paragraph
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.baselines import fit_age_sex
from src.data import build_sequences, load_split, load_vocab
from src.evaluate import evaluate
from src.model import HealthTransformer, make_predict_fn, select_device

OUTPUTS_DIR = Path("outputs")
CHECKPOINT_PATH = Path("checkpoints/best.pt")
_EXCLUDED = 402867
_BATCH = 256
_P_FLOOR = 1e-12       # probability floor for log(); same clip evaluate.py applies to NLL
_SHUFFLE_SEED = 0      # seed for the within-patient permutation in analysis 3
_NLL_TOL = 1e-4        # max allowed gap between recomputed and saved age_sex NLL
_LEAK_TOL = 1e-5       # max allowed |logit diff| in the causal-mask leakage test


# ---------------------------------------------------------------------------
# Shared helper: prefix-by-prefix NLL, same protocol as evaluate.py
# ---------------------------------------------------------------------------

def _mean_nll(
    predict_fn: Callable,
    sequences: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Mean NLL over all prediction points using predict_fn."""
    nll_sum, n = 0.0, 0
    for i in range(0, len(sequences), _BATCH):
        batch = sequences[i : i + _BATCH]
        prefixes: list[tuple[np.ndarray, np.ndarray]] = []
        true_toks: list[int] = []
        for tokens, ages in batch:
            for k in range(1, len(tokens)):
                prefixes.append((tokens[:k], ages[:k]))
                true_toks.append(int(tokens[k]))
        if not prefixes:
            continue
        probs = predict_fn(prefixes)
        t = np.array(true_toks, dtype=np.int64)
        tp = np.clip(probs[np.arange(len(t)), t], _P_FLOOR, None)
        nll_sum += float(-np.log(tp).sum())
        n += len(t)
    return nll_sum / n if n else float("nan")


# ---------------------------------------------------------------------------
# 1. Per-disease paired comparison
# ---------------------------------------------------------------------------

def run_per_disease_paired() -> dict:
    """Join per-disease AUC CSVs, report paired differences, save scatter."""
    trans = pd.read_csv(OUTPUTS_DIR / "transformer_per_disease_auc.csv")
    base = pd.read_csv(OUTPUTS_DIR / "age_sex_per_disease_auc.csv")

    j = trans.merge(base, on="disease", suffixes=("_transformer", "_age_sex"))
    j["diff"] = j["auc_transformer"] - j["auc_age_sex"]

    frac = float((j["diff"] > 0).mean())
    mean_d = float(j["diff"].mean())
    med_d = float(j["diff"].median())

    out = (
        j[["disease", "auc_transformer", "auc_age_sex", "diff", "n_positives_transformer"]]
        .rename(columns={"n_positives_transformer": "n_positives"})
        .sort_values("diff", ascending=False)
    )
    out.to_csv(OUTPUTS_DIR / "per_disease_paired.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        j["auc_age_sex"],
        j["auc_transformer"],
        c=np.log10(j["n_positives_transformer"].clip(lower=1)),
        cmap="viridis",
        s=12,
        alpha=0.7,
    )
    lo = min(j["auc_age_sex"].min(), j["auc_transformer"].min()) - 0.02
    hi = max(j["auc_age_sex"].max(), j["auc_transformer"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
    plt.colorbar(sc, ax=ax, label="log₁₀(n_positives)")
    ax.set_xlabel("age_sex AUROC")
    ax.set_ylabel("Transformer AUROC")
    ax.set_title(f"Per-disease AUROC  (n = {len(j)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "per_disease_scatter.png", dpi=150)
    plt.close(fig)

    print(
        f"\n[1] Per-disease paired  n={len(j)}"
        f"  transformer > age_sex: {frac:.1%} ({(j['diff'] > 0).sum()}/{len(j)})"
        f"  mean diff: {mean_d:+.4f}  median: {med_d:+.4f}"
    )
    return {"n_diseases": len(j), "frac_better": frac, "mean_diff": mean_d, "median_diff": med_d}


# ---------------------------------------------------------------------------
# 2. Stratified comparison
# ---------------------------------------------------------------------------

def run_stratified_comparison() -> dict:
    """Mean AUROC per stratum for both models; bar chart."""
    t_s = pd.read_csv(OUTPUTS_DIR / "transformer_stratified_auc.csv")
    b_s = pd.read_csv(OUTPUTS_DIR / "age_sex_stratified_auc.csv")

    t_g = (
        t_s.groupby("stratum")
        .agg(transformer_mean_auc=("auc", "mean"), n_diseases=("disease", "nunique"))
        .reset_index()
    )
    b_g = (
        b_s.groupby("stratum")
        .agg(age_sex_mean_auc=("auc", "mean"))
        .reset_index()
    )
    merged = t_g.merge(b_g, on="stratum")

    order = ["Female", "Male", "age<40", "age40-60", "age>60"]
    merged["_ord"] = merged["stratum"].map({s: i for i, s in enumerate(order)})
    merged = merged.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    merged.to_csv(OUTPUTS_DIR / "stratified_comparison.csv", index=False)

    x = np.arange(len(merged))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, merged["transformer_mean_auc"], w, label="transformer")
    ax.bar(x + w / 2, merged["age_sex_mean_auc"], w, label="age_sex")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["stratum"])
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Mean per-disease AUROC by stratum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "stratified_bar.png", dpi=150)
    plt.close(fig)

    print(f"\n[2] Stratified comparison:")
    for _, row in merged.iterrows():
        print(
            f"    {row['stratum']:<12}"
            f"  transformer={row['transformer_mean_auc']:.4f}"
            f"  age_sex={row['age_sex_mean_auc']:.4f}"
            f"  n_diseases={row['n_diseases']}"
        )
    return {"stratified": merged.to_dict("records")}


# ---------------------------------------------------------------------------
# 3. Shuffled-history test
# ---------------------------------------------------------------------------

def run_shuffled_history(
    transformer_fn: Callable,
    age_sex_fn: Callable,
    val_sequences: list[tuple[np.ndarray, np.ndarray]],
    vocab: list[str],
    unseen: set[int],
) -> dict:
    """Permute diagnosis tokens within each patient; score transformer and age_sex on the same shuffle.

    Ages and the sex token stay in place, so the age_sex baseline's prediction for each
    prefix is unchanged; only its targets move.  Its drop therefore isolates the effect of
    breaking the age-to-diagnosis pairing and is the control for the transformer's drop.
    The recomputed unshuffled age_sex NLL is checked against baselines_summary.csv.

    Writes, via evaluate(): transformer_shuffled_{per_disease,stratified}_auc.csv and
    age_sex_shuffled_{per_disease,stratified}_auc.csv.
    """
    # Unshuffled reference numbers.  NLL is recomputed because the transformer's val NLL is
    # only printed by run_eval.py, never saved; age_sex is recomputed the same way for
    # symmetry.  Mean AUROC is read from the per-disease CSVs, which is exactly how
    # evaluate() defines mean_auroc.
    t_nll_u = _mean_nll(transformer_fn, val_sequences)
    a_nll_u = _mean_nll(age_sex_fn, val_sequences)
    t_auc_u = float(pd.read_csv(OUTPUTS_DIR / "transformer_per_disease_auc.csv")["auc"].mean())
    a_auc_u = float(pd.read_csv(OUTPUTS_DIR / "age_sex_per_disease_auc.csv")["auc"].mean())

    # Guard: the age_sex baseline refitted here must be the one baselines.py scored.
    # Same train set, same K and same protocol => the two NLLs should agree exactly.
    ref_df = pd.read_csv(OUTPUTS_DIR / "baselines_summary.csv")
    ref_nll = float(ref_df.loc[ref_df["name"] == "age_sex", "mean_nll"].iloc[0])
    if abs(a_nll_u - ref_nll) > _NLL_TOL:
        raise RuntimeError(
            f"age_sex NLL mismatch: recomputed {a_nll_u:.6f} vs "
            f"baselines_summary.csv {ref_nll:.6f} (tolerance {_NLL_TOL})"
        )

    # One permutation per patient, built once and shared by both models.
    rng = np.random.default_rng(_SHUFFLE_SEED)
    shuffled: list[tuple[np.ndarray, np.ndarray]] = []
    for tokens, ages in val_sequences:
        new_tok = tokens.copy()
        new_tok[1:] = rng.permutation(tokens[1:])  # sex token fixed at 0; ages unchanged
        shuffled.append((new_tok, ages))

    t_s = evaluate(
        transformer_fn, shuffled, vocab, "transformer_shuffled",
        excluded_from_auroc=unseen,
    )
    a_s = evaluate(
        age_sex_fn, shuffled, vocab, "age_sex_shuffled",
        excluded_from_auroc=unseen,
    )

    out = {
        "shuffle_seed": _SHUFFLE_SEED,
        "transformer_unshuffled_nll": t_nll_u,
        "transformer_shuffled_nll": t_s["mean_nll"],
        "transformer_nll_delta": t_s["mean_nll"] - t_nll_u,
        "transformer_unshuffled_auroc": t_auc_u,
        "transformer_shuffled_auroc": t_s["mean_auroc"],
        "transformer_auroc_delta": t_s["mean_auroc"] - t_auc_u,
        "age_sex_unshuffled_nll": a_nll_u,
        "age_sex_shuffled_nll": a_s["mean_nll"],
        "age_sex_nll_delta": a_s["mean_nll"] - a_nll_u,
        "age_sex_unshuffled_auroc": a_auc_u,
        "age_sex_shuffled_auroc": a_s["mean_auroc"],
        "age_sex_auroc_delta": a_s["mean_auroc"] - a_auc_u,
    }

    print(f"\n[3] Shuffled-history test  (shuffle seed = {_SHUFFLE_SEED}, same permutation for both models)")
    print(
        f"    age_sex NLL check: recomputed {a_nll_u:.6f}  baselines_summary.csv {ref_nll:.6f}"
        f"  |diff| = {abs(a_nll_u - ref_nll):.2e} <= {_NLL_TOL} OK"
    )
    print(f"    {'model':<12} {'NLL unshuf':>11} {'NLL shuf':>10} {'ΔNLL':>8}   {'AUROC unshuf':>13} {'AUROC shuf':>11} {'ΔAUROC':>8}")
    for m in ("transformer", "age_sex"):
        print(
            f"    {m:<12} {out[f'{m}_unshuffled_nll']:>11.4f} {out[f'{m}_shuffled_nll']:>10.4f}"
            f" {out[f'{m}_nll_delta']:>+8.4f}   {out[f'{m}_unshuffled_auroc']:>13.4f}"
            f" {out[f'{m}_shuffled_auroc']:>11.4f} {out[f'{m}_auroc_delta']:>+8.4f}"
        )
    print(
        "    Interpretation: age_sex sees the same shuffle, so its drop measures how much of the\n"
        "    change comes from breaking the age-to-diagnosis pairing alone. The transformer's drop\n"
        "    beyond that is attributable to temporal structure in the history (order and age\n"
        "    alignment), beyond which diagnoses co-occur in the patient. Shuffled prefixes still\n"
        "    contain diagnoses from the patient's future, so co-occurrence survives the shuffle."
    )
    return out


# ---------------------------------------------------------------------------
# 4. Leakage test
# ---------------------------------------------------------------------------

def run_leakage_test(
    model: HealthTransformer,
    vocab: list[str],
    val_sequences: list[tuple[np.ndarray, np.ndarray]],
    n_patients: int = 50,
) -> dict:
    """Assert the causal mask: nothing after position k-1 can change the logits at k-1.

    Runs on CPU on a deep copy of the model, so the shared model behind transformer_fn
    stays on its own device.  For each of n_patients val patients with at least 4 events,
    three positions are tested: early k=1, middle k=L//2, late k=L-1.  Positions 0..k-1 are
    held fixed while tokens[k:L] are replaced with random valid diagnoses and ages[k:L]
    with random strictly increasing ages above age[k-1].  Logits at k-1 must agree to
    within _LEAK_TOL.
    """
    cpu = torch.device("cpu")
    m = copy.deepcopy(model).to(cpu)  # copy: transformer_fn still holds the original on its device
    m.eval()                          # dropout off, otherwise the two passes differ by design
    BLOCK = HealthTransformer.BLOCK_SIZE

    # Valid diagnosis tokens: all vocab indices except padding (0) and sex tokens
    sex_names = {"Female", "Male"}
    valid_diag = np.array(
        [i for i, v in enumerate(vocab) if v not in sex_names and i != 0],
        dtype=np.int64,
    )

    rng = np.random.default_rng(0)
    max_diffs: list[float] = []
    tested_patients = 0

    for tokens, ages in val_sequences:
        if tested_patients >= n_patients:
            break
        L = len(tokens)
        if L < 4:
            continue  # need 1 < L//2 < L-1 for three distinct positions
        tested_patients += 1

        # Original sequence, padded to BLOCK_SIZE; one forward pass serves all three k
        tok1 = np.zeros(BLOCK, dtype=np.int64)
        age1 = np.zeros(BLOCK, dtype=np.float32)
        tok1[:L] = tokens[:L]
        age1[:L] = ages[:L] / 365.25  # days → years
        tok1_t = torch.from_numpy(tok1[None])
        age1_t = torch.from_numpy(age1[None])
        with torch.no_grad():
            logits1 = m(tok1_t, age1_t, key_padding_mask=(tok1_t == 0))[0].numpy()  # (BLOCK, V)

        for k in (1, L // 2, L - 1):
            # Identical prefix [0..k-1]; tokens[k:L] → random valid diagnoses;
            # ages[k:L] → random strictly increasing ages > age at position k-1.
            tok2 = tok1.copy()
            age2 = age1.copy()
            tok2[k:L] = rng.choice(valid_diag, size=L - k, replace=True)
            base_age_yrs = float(age1[k - 1])
            deltas = rng.uniform(0.1, 5.0, size=L - k).astype(np.float32)
            age2[k:L] = (base_age_yrs + np.cumsum(deltas)).astype(np.float32)

            tok2_t = torch.from_numpy(tok2[None])
            age2_t = torch.from_numpy(age2[None])
            with torch.no_grad():
                l2 = m(tok2_t, age2_t, key_padding_mask=(tok2_t == 0))[0, k - 1].numpy()

            max_diffs.append(float(np.abs(logits1[k - 1] - l2).max()))

    if tested_patients < n_patients:
        raise RuntimeError(
            f"leakage test: only {tested_patients} val patients with >= 4 events, needed {n_patients}"
        )

    n_cases = len(max_diffs)
    overall_max = float(max(max_diffs))
    status = "PASS" if overall_max <= _LEAK_TOL else "FAIL"
    print(
        f"\n[4] Leakage test: {status}"
        f"  max |logit diff| = {overall_max:.2e}"
        f"  ({n_cases} cases = {tested_patients} patients × 3 positions [k=1, L//2, L-1],"
        f" tokens and ages after k perturbed, CPU)"
    )
    return {
        "status": status,
        "max_diff": overall_max,
        "n_patients": tested_patients,
        "n_cases": n_cases,
        "device": "cpu",
    }


# ---------------------------------------------------------------------------
# 5. History-length curve
# ---------------------------------------------------------------------------

def run_history_length(
    transformer_fn: Callable,
    age_sex_fn: Callable,
    val_sequences: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """NLL and top-20 accuracy bucketed by prefix length for both models."""
    buckets = [("1–4", 1, 4), ("5–9", 5, 9), ("10–19", 10, 19), ("20–29", 20, 29), ("30+", 30, 10**9)]
    b_labels = [b[0] for b in buckets]

    stats: dict[str, list[dict]] = {
        m: [{"nll_sum": 0.0, "top20": 0.0, "count": 0.0} for _ in buckets]
        for m in ("transformer", "age_sex")
    }

    for bi in range(0, len(val_sequences), _BATCH):
        batch = val_sequences[bi : bi + _BATCH]
        prefixes: list[tuple[np.ndarray, np.ndarray]] = []
        true_toks: list[int] = []
        lens: list[int] = []

        for tokens, ages in batch:
            for k in range(1, len(tokens)):
                prefixes.append((tokens[:k], ages[:k]))
                true_toks.append(int(tokens[k]))
                lens.append(k)  # prefix length = k (tokens 0..k-1 seen)

        if not prefixes:
            continue

        true_arr = np.array(true_toks, dtype=np.int64)
        lens_arr = np.array(lens, dtype=np.int32)

        for m_name, fn in (("transformer", transformer_fn), ("age_sex", age_sex_fn)):
            probs = fn(prefixes)
            tp = np.clip(probs[np.arange(len(true_arr)), true_arr], 1e-12, None)
            nlls = -np.log(tp)
            top20_idx = np.argpartition(probs, -20, axis=1)[:, -20:]
            hits = (top20_idx == true_arr[:, np.newaxis]).any(axis=1).astype(float)

            for b_idx, (_, lo, hi) in enumerate(buckets):
                mask = (lens_arr >= lo) & (lens_arr <= hi)
                stats[m_name][b_idx]["nll_sum"] += float(nlls[mask].sum())
                stats[m_name][b_idx]["top20"] += float(hits[mask].sum())
                stats[m_name][b_idx]["count"] += float(mask.sum())

    rows: list[dict] = []
    for m in ("transformer", "age_sex"):
        for b_idx, (label, _, __) in enumerate(buckets):
            s = stats[m][b_idx]
            n = s["count"]
            rows.append({
                "bucket": label,
                "model": m,
                "mean_nll": s["nll_sum"] / n if n else float("nan"),
                "top20_acc": s["top20"] / n if n else float("nan"),
                "n_points": int(n),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS_DIR / "history_length.csv", index=False)

    x_pos = np.arange(len(buckets))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for m, style in (("transformer", "-o"), ("age_sex", "--s")):
        sub = df[df["model"] == m].reset_index(drop=True)
        ax1.plot(x_pos, sub["mean_nll"], style, label=m)
        ax2.plot(x_pos, sub["top20_acc"], style, label=m)
    for ax in (ax1, ax2):
        ax.set_xticks(x_pos)
        ax.set_xticklabels(b_labels)
    ax1.set_ylabel("Mean NLL (nats)")
    ax1.set_title("Performance by history length")
    ax1.legend()
    ax2.set_ylabel("Top-20 accuracy")
    ax2.set_xlabel("Prefix length (events seen before prediction)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "history_length.png", dpi=150)
    plt.close(fig)

    print(f"\n[5] History-length curve:")
    for m in ("transformer", "age_sex"):
        for _, row in df[df["model"] == m].iterrows():
            print(
                f"    {m:<12}  bucket={row['bucket']:<6}"
                f"  NLL={row['mean_nll']:.4f}  top20={row['top20_acc']:.4f}"
                f"  n={int(row['n_points']):,}"
            )
    return {"history_df": df}


# ---------------------------------------------------------------------------
# 6. Calibration
# ---------------------------------------------------------------------------

def run_calibration(
    transformer_fn: Callable,
    val_sequences: list[tuple[np.ndarray, np.ndarray]],
    vocab: list[str],
) -> dict:
    """Pooled calibration and per-disease observed/expected over the AUROC-eligible diseases.

    The disease set is whatever transformer_per_disease_auc.csv lists, i.e. the eligible
    set chosen by evaluate.py; its size is derived from that file, not fixed.  Every
    (prediction point, eligible disease) pair contributes one (predicted probability,
    binary label).  Pairs are binned by decade of predicted probability (fixed log-spaced
    edges at powers of 10 spanning the observed range) and ECE is the count-weighted mean
    |mean predicted − observed rate| over bins.  Per disease, O/E = positives / sum of
    predicted probability; O/E > 1 means the model under-predicts that disease overall.

    Writes calibration.csv (per bin), calibration_per_disease.csv (per disease) and
    calibration.png.
    """
    elig_df = pd.read_csv(OUTPUTS_DIR / "transformer_per_disease_auc.csv")
    eligible_names = set(elig_df["disease"])
    eligible_list = [i for i, v in enumerate(vocab) if v in eligible_names]
    eligible_arr = np.array(eligible_list, dtype=np.int64)
    n_elig = len(eligible_list)

    # Collect per-point probs for eligible diseases and true tokens
    probs_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []

    for bi in range(0, len(val_sequences), _BATCH):
        batch = val_sequences[bi : bi + _BATCH]
        prefixes: list[tuple[np.ndarray, np.ndarray]] = []
        true_toks: list[int] = []
        for tokens, ages in batch:
            for k in range(1, len(tokens)):
                prefixes.append((tokens[:k], ages[:k]))
                true_toks.append(int(tokens[k]))
        if not prefixes:
            continue
        probs = transformer_fn(prefixes)  # (n, vocab)
        probs_chunks.append(probs[:, eligible_arr].astype(np.float32))
        true_chunks.append(np.array(true_toks, dtype=np.int64))

    probs_arr = np.vstack(probs_chunks)       # (n_pts, n_elig)
    true_arr = np.concatenate(true_chunks)    # (n_pts,)
    n_pts = len(true_arr)

    # Fixed log-spaced bins: one per decade, edges at powers of 10 covering [p_min, p_max].
    # _P_FLOOR guards against float32 softmax underflow to exactly 0 (log10(0) undefined).
    p_min = max(float(probs_arr.min()), _P_FLOOR)
    p_max = float(probs_arr.max())
    lo_exp = int(np.floor(np.log10(p_min)))
    hi_exp = int(np.ceil(np.log10(p_max)))       # probabilities < 1 so this is <= 0
    edges = np.logspace(lo_exp, hi_exp, hi_exp - lo_exp + 1)
    n_bins = len(edges) - 1
    if n_bins < 1:
        raise RuntimeError(f"calibration: degenerate probability range [{p_min:.3e}, {p_max:.3e}]")

    # Accumulate bin sums disease by disease to avoid materialising the full
    # (n_pts × n_elig) label matrix at once; per-disease O/E falls out of the same pass.
    mean_pred = np.zeros(n_bins, dtype=np.float64)
    mean_obs = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.float64)
    oe_rows: list[dict] = []

    for d_idx in range(n_elig):
        d_tok = eligible_list[d_idx]
        d_probs = probs_arr[:, d_idx].astype(np.float64)     # (n_pts,)
        d_labels = (true_arr == d_tok).astype(np.float64)   # (n_pts,)
        # digitize against inner edges: below edges[1] → bin 0, at/above edges[-2] → bin n_bins-1,
        # so every pair lands in exactly one bin and counts sum to n_pts * n_elig
        bids = np.digitize(d_probs, edges[1:-1])
        mean_pred += np.bincount(bids, weights=d_probs, minlength=n_bins)
        mean_obs += np.bincount(bids, weights=d_labels, minlength=n_bins)
        bin_counts += np.bincount(bids, minlength=n_bins)

        n_pos = float(d_labels.sum())      # observed: times disease d was the next event
        sum_pred = float(d_probs.sum())    # expected: total probability placed on d
        oe_rows.append({
            "disease": vocab[d_tok],
            "n_positives": int(n_pos),
            "sum_predicted": sum_pred,
            "o_e_ratio": n_pos / sum_pred,  # sum_pred > 0: softmax probs are strictly positive
        })

    # Cross-check: positives counted here must match n_positives in the AUROC CSV
    oe_df = pd.DataFrame(oe_rows)
    chk = oe_df.merge(elig_df[["disease", "n_positives"]], on="disease", suffixes=("", "_auc_csv"))
    bad = chk[chk["n_positives"] != chk["n_positives_auc_csv"]]
    if len(chk) != n_elig or len(bad):
        raise RuntimeError(
            f"calibration: n_positives mismatch vs transformer_per_disease_auc.csv for "
            f"{len(bad)} diseases (matched {len(chk)} of {n_elig})"
        )

    oe_df = oe_df.sort_values("o_e_ratio", ascending=False).reset_index(drop=True)
    oe_df.to_csv(OUTPUTS_DIR / "calibration_per_disease.csv", index=False)
    oe = oe_df["o_e_ratio"].to_numpy()
    oe_q25, oe_med, oe_q75 = (float(q) for q in np.percentile(oe, [25, 50, 75]))

    # Pooled ECE
    nonzero = bin_counts > 0
    mean_pred[nonzero] /= bin_counts[nonzero]
    mean_obs[nonzero] /= bin_counts[nonzero]
    weights = bin_counts / bin_counts.sum()
    ece = float(np.abs(mean_pred - mean_obs) @ weights)  # empty bins have weight 0

    cal_df = pd.DataFrame({
        "bin": np.arange(n_bins),
        "lower_edge": edges[:-1],
        "upper_edge": edges[1:],
        "mean_predicted": np.where(nonzero, mean_pred, np.nan),  # NaN, not 0, for empty bins
        "observed_rate": np.where(nonzero, mean_obs, np.nan),
        "count": bin_counts.astype(np.int64),
    })
    cal_df.to_csv(OUTPUTS_DIR / "calibration.csv", index=False)

    # Plot on log-log axes; bins with observed rate exactly 0 cannot be drawn on a log
    # axis and are omitted from the figure (they remain in calibration.csv)
    plot_mask = nonzero & (mean_obs > 0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        mean_pred[plot_mask],
        mean_obs[plot_mask],
        s=20 + weights[plot_mask] * 2000,
        alpha=0.8,
        zorder=3,
    )
    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], "k--", lw=0.8, label="perfect calibration")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted probability (per decade bin)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration  (ECE = {ece:.4f}, {n_bins} log bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "calibration.png", dpi=150)
    plt.close(fig)

    total_pairs = n_pts * n_elig
    print(
        f"\n[6] Calibration  ECE={ece:.4f}  n_bins={n_bins} (decades 1e{lo_exp}..1e{hi_exp})"
        f"  n_diseases={n_elig}  n_points={n_pts:,}  n_pairs={total_pairs:,}"
    )
    print(f"    per-disease O/E: median={oe_med:.4f}  IQR=[{oe_q25:.4f}, {oe_q75:.4f}]")
    return {
        "ece": ece,
        "n_bins": n_bins,
        "n_diseases": n_elig,
        "n_points": n_pts,
        "n_pairs": total_pairs,
        "o_e_median": oe_med,
        "o_e_q25": oe_q25,
        "o_e_q75": oe_q75,
        "o_e_iqr": oe_q75 - oe_q25,
    }


# ---------------------------------------------------------------------------
# 7. Failure analysis
# ---------------------------------------------------------------------------

def run_failure_analysis() -> dict:
    """Bottom-15 diseases by AUROC and by transformer–age_sex gap; scatter vs prevalence."""
    trans = pd.read_csv(OUTPUTS_DIR / "transformer_per_disease_auc.csv")
    base = pd.read_csv(OUTPUTS_DIR / "age_sex_per_disease_auc.csv")

    j = trans.merge(base, on="disease", suffixes=("_transformer", "_age_sex"))
    j["diff"] = j["auc_transformer"] - j["auc_age_sex"]

    low_auc = set(j.nsmallest(15, "auc_transformer")["disease"])
    low_gap = set(j.nsmallest(15, "diff")["disease"])

    def _ftype(d: str) -> str:
        if d in low_auc and d in low_gap:
            return "both"
        if d in low_auc:
            return "low_auc"
        return "underperforms_age_sex"

    failure_diseases = low_auc | low_gap
    failures = j[j["disease"].isin(failure_diseases)][
        ["disease", "auc_transformer", "auc_age_sex", "diff", "n_positives_transformer"]
    ].rename(columns={"n_positives_transformer": "n_positives"}).copy()
    failures["failure_type"] = failures["disease"].map(_ftype)
    failures = failures.sort_values("auc_transformer").reset_index(drop=True)
    failures.to_csv(OUTPUTS_DIR / "failures.csv", index=False)

    # Scatter: transformer AUROC vs log10(n_positives)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        np.log10(j["n_positives_transformer"].clip(lower=1)),
        j["auc_transformer"],
        s=8,
        alpha=0.35,
        color="steelblue",
        label="all diseases",
    )
    palette = {"low_auc": ("red", "v"), "underperforms_age_sex": ("orange", "s"), "both": ("purple", "*")}
    for ftype, (color, marker) in palette.items():
        sub = failures[failures["failure_type"] == ftype]
        if len(sub):
            ax.scatter(
                np.log10(sub["n_positives"].clip(lower=1)),
                sub["auc_transformer"],
                s=55,
                color=color,
                marker=marker,
                label=ftype,
                zorder=5,
            )
    ax.set_xlabel("log₁₀(n_positives)")
    ax.set_ylabel("Transformer AUROC")
    ax.set_title("Transformer AUROC vs disease prevalence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "auc_vs_prevalence.png", dpi=150)
    plt.close(fig)

    min_auc = float(j.nsmallest(15, "auc_transformer")["auc_transformer"].max())  # worst of bottom-15
    worst_gap = float(j.nsmallest(15, "diff")["diff"].min())
    print(
        f"\n[7] Failure analysis  {len(failures)} unique failure-mode diseases"
        f"  (low_auc={len(low_auc - low_gap)}, underperforms={len(low_gap - low_auc)}, both={len(low_auc & low_gap)})"
    )
    print(f"    Bottom-15 by AUROC: max in set = {min_auc:.4f}")
    print(f"    Bottom-15 by gap:   worst diff = {worst_gap:+.4f}")
    return {"n_failures": len(failures), "min_auc": min_auc, "worst_gap": worst_gap}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Shared data
    train_df = load_split("train")
    val_df = load_split("val")
    vocab = load_vocab()
    train_df = train_df[train_df["person_id"] != _EXCLUDED]
    train_sequences = build_sequences(train_df)
    val_sequences = build_sequences(val_df)
    unseen: set[int] = set(val_df["token"]) - set(train_df["token"])
    sex_indices = {i for i, v in enumerate(vocab) if v in ("Female", "Male")}

    # Transformer
    device = select_device()
    print(f"Device: {device}")
    model = HealthTransformer()
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.to(device)
    transformer_fn = make_predict_fn(model, device, vocab)

    # Age-sex baseline refitted with the K baselines.py chose for it on dev NLL.
    # Filter to the age_sex rows explicitly before taking the argmin: both baselines
    # may happen to share the same best K, so an unfiltered argmin could pass the NLL
    # check in run_shuffled_history by coincidence.
    tuning_df = pd.read_csv(OUTPUTS_DIR / "baseline_K_tuning.csv")
    age_sex_rows = tuning_df[tuning_df["baseline"] == "age_sex"]
    if age_sex_rows.empty:
        raise RuntimeError("baseline_K_tuning.csv has no rows with baseline == 'age_sex'")
    best_K = int(age_sex_rows.loc[age_sex_rows["dev_nll"].idxmin(), "K"])
    age_sex_fn = fit_age_sex(train_sequences, vocab, sex_indices, K=float(best_K))
    print(f"age_sex refitted with K={best_K} (argmin dev_nll over age_sex rows of baseline_K_tuning.csv)")

    # Run all seven analyses
    r1 = run_per_disease_paired()
    r2 = run_stratified_comparison()
    r3 = run_shuffled_history(transformer_fn, age_sex_fn, val_sequences, vocab, unseen)
    r4 = run_leakage_test(model, vocab, val_sequences)
    r5 = run_history_length(transformer_fn, age_sex_fn, val_sequences)
    r6 = run_calibration(transformer_fn, val_sequences, vocab)
    r7 = run_failure_analysis()

    hl = r5["history_df"]
    short_top20 = float(hl[(hl["model"] == "transformer") & (hl["bucket"] == "1–4")]["top20_acc"].iloc[0])
    long_top20 = float(hl[(hl["model"] == "transformer") & (hl["bucket"] == "30+")]["top20_acc"].iloc[0])

    # Single source of truth for the printed paragraph: every value cast explicitly,
    # written to JSON first, then the paragraph is formatted from the same dict.
    summary = {
        "provenance": {
            "checkpoint": str(CHECKPOINT_PATH),
            "device": str(device),
            "age_sex_K": int(best_K),
            "shuffle_seed": int(_SHUFFLE_SEED),
            "n_val_patients": int(len(val_sequences)),
        },
        "per_disease": {
            "n_diseases": int(r1["n_diseases"]),
            "frac_better": float(r1["frac_better"]),
            "mean_diff": float(r1["mean_diff"]),
            "median_diff": float(r1["median_diff"]),
        },
        "shuffle": {
            k: (int(v) if k == "shuffle_seed" else float(v)) for k, v in r3.items()
        },
        "leakage": {
            "status": str(r4["status"]),
            "max_diff": float(r4["max_diff"]),
            "n_patients": int(r4["n_patients"]),
            "n_cases": int(r4["n_cases"]),
            "device": str(r4["device"]),
        },
        "history_length": {
            "transformer_top20_prefix_1_4": float(short_top20),
            "transformer_top20_prefix_30plus": float(long_top20),
        },
        "calibration": {
            "ece": float(r6["ece"]),
            "n_bins": int(r6["n_bins"]),
            "n_diseases": int(r6["n_diseases"]),
            "n_points": int(r6["n_points"]),
            "n_pairs": int(r6["n_pairs"]),
            "o_e_median": float(r6["o_e_median"]),
            "o_e_q25": float(r6["o_e_q25"]),
            "o_e_q75": float(r6["o_e_q75"]),
            "o_e_iqr": float(r6["o_e_iqr"]),
        },
        "failures": {
            "n_failures": int(r7["n_failures"]),
            "min_auc": float(r7["min_auc"]),
            "worst_gap": float(r7["worst_gap"]),
        },
    }
    with open(OUTPUTS_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUTPUTS_DIR}/analysis_summary.json")

    s = summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print(
        f"The transformer outperforms age_sex on {s['per_disease']['frac_better']:.1%} of "
        f"{s['per_disease']['n_diseases']} diseases (mean AUROC gap {s['per_disease']['mean_diff']:+.3f}, "
        f"median {s['per_disease']['median_diff']:+.3f}). "
        f"Shuffling diagnosis order within each patient (seed {s['shuffle']['shuffle_seed']}) changes "
        f"transformer NLL by {s['shuffle']['transformer_nll_delta']:+.3f} nats and mean AUROC by "
        f"{s['shuffle']['transformer_auroc_delta']:+.3f}; the same shuffle changes age_sex NLL by "
        f"{s['shuffle']['age_sex_nll_delta']:+.3f} and AUROC by {s['shuffle']['age_sex_auroc_delta']:+.3f} "
        f"(unshuffled age_sex AUROC {s['shuffle']['age_sex_unshuffled_auroc']:.3f}). "
        f"The causal-mask leakage test {s['leakage']['status']} (max |logit diff| "
        f"{s['leakage']['max_diff']:.1e} over {s['leakage']['n_cases']} cases: "
        f"{s['leakage']['n_patients']} val patients × 3 positions, CPU). "
        f"Top-20 accuracy improves with prefix length: "
        f"{s['history_length']['transformer_top20_prefix_1_4']:.3f} for 1–4 events seen vs "
        f"{s['history_length']['transformer_top20_prefix_30plus']:.3f} for 30+ events seen. "
        f"Per-disease observed/expected ratios have median {s['calibration']['o_e_median']:.3f} "
        f"(IQR {s['calibration']['o_e_q25']:.3f}–{s['calibration']['o_e_q75']:.3f}) over "
        f"{s['calibration']['n_diseases']} diseases. "
        f"{s['failures']['n_failures']} diseases are identified as failure modes (lowest transformer AUROC "
        f"in bottom-15: up to {s['failures']['min_auc']:.3f}; worst gap vs age_sex: "
        f"{s['failures']['worst_gap']:+.3f})."
    )
    print("=" * 72)
