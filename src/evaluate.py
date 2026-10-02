"""Fixed evaluation protocol for health trajectory models.

This module defines the canonical scoring function. It must never be changed
after baselines are established, so that all models are compared on identical
ground.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUTPUTS_DIR = Path("outputs")
_BATCH_SIZE = 256


def evaluate(
    predict_fn: Callable[[list[tuple[np.ndarray, np.ndarray]]], np.ndarray],
    sequences: list[tuple[np.ndarray, np.ndarray]],
    vocab: list[str],
    name: str,
    *,
    excluded_from_auroc: set[int] | None = None,
    batch_size: int = _BATCH_SIZE,
) -> dict:
    """Evaluate a predict_fn on sequences and return a metrics dict.

    Fixed prediction protocol:
    - Prediction points: k = 1..L-1 per patient.
    - Input to predict_fn at point k: tokens t_0..t_{k-1} and ages a_0..a_{k-1}.
    - a_k (age of the target event) is never passed; passing it would leak the future.
    - excluded_from_auroc tokens count against cross-entropy but are excluded from
      per-disease AUROC (used to exclude val tokens absent from train).

    Args:
        predict_fn: takes list of (tokens, ages) prefix pairs, returns float array
                    of shape (n_prefixes, vocab_size) with rows summing to 1.
        sequences:  output of build_sequences(val_df).
        vocab:      output of load_vocab().
        name:       prefix for output CSV filenames.
        excluded_from_auroc: token indices to skip in per-disease AUROC.
        batch_size: patients per batch; controls peak memory of predict_fn call.

    Returns:
        Dict with keys: name, mean_nll, top1_acc, top5_acc, top20_acc,
        mean_auroc, median_auroc, n_diseases_auroc, n_points.

    Saves:
        outputs/{name}_per_disease_auc.csv
        outputs/{name}_stratified_auc.csv
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    vocab_size = len(vocab)
    sex_indices = {i for i, v in enumerate(vocab) if v in ("Female", "Male")}
    female_tok = next((i for i, v in enumerate(vocab) if v == "Female"), None)
    male_tok = next((i for i, v in enumerate(vocab) if v == "Male"), None)
    excluded = excluded_from_auroc or set()

    # --- Pre-scan: count how often each token is a prediction target ---
    target_counts: Counter[int] = Counter()
    for tokens, _ in sequences:
        for tok in tokens[1:]:
            target_counts[tok] += 1

    eligible_list: list[int] = sorted(
        tok
        for tok, cnt in target_counts.items()
        if cnt >= 20 and tok not in sex_indices and tok not in excluded
    )
    eligible_arr = np.array(eligible_list, dtype=np.int64)
    n_eligible = len(eligible_list)

    # --- Main pass: batch over patients ---
    nll_sum = 0.0
    n_points = 0
    top_k_hits: dict[int, int] = {1: 0, 5: 0, 20: 0}

    true_tokens_chunks: list[np.ndarray] = []
    sex_chunks: list[np.ndarray] = []
    age_band_chunks: list[np.ndarray] = []
    disease_probs_chunks: list[np.ndarray] = []

    for batch_start in range(0, len(sequences), batch_size):
        batch = sequences[batch_start : batch_start + batch_size]

        prefixes: list[tuple[np.ndarray, np.ndarray]] = []
        true_toks_batch: list[int] = []
        sex_batch: list[int] = []
        age_band_batch: list[int] = []

        for tokens, ages in batch:
            if len(tokens) < 2:
                continue
            sex_tok = int(tokens[0])
            for k in range(1, len(tokens)):
                prefixes.append((tokens[:k], ages[:k]))
                true_toks_batch.append(int(tokens[k]))
                sex_batch.append(sex_tok)
                cur_age_yrs = float(ages[k - 1]) / 365.25
                age_band_batch.append(0 if cur_age_yrs < 40.0 else 1 if cur_age_yrs < 60.0 else 2)

        if not prefixes:
            continue

        probs = predict_fn(prefixes)  # (n, vocab_size) — full array, used then discarded

        true_arr = np.array(true_toks_batch, dtype=np.int64)
        n = len(prefixes)

        # NLL (natural log; clip for numerical safety)
        true_probs = np.clip(probs[np.arange(n), true_arr], 1e-12, None)
        nll_sum += float(-np.log(true_probs).sum())
        n_points += n

        # Top-k accuracy
        for k in [1, 5, 20]:
            topk_idx = np.argpartition(probs, -k, axis=1)[:, -k:]
            hits = (topk_idx == true_arr[:, np.newaxis]).any(axis=1)
            top_k_hits[k] += int(hits.sum())

        # Retain only eligible disease columns; discard the full probs array
        if n_eligible > 0:
            disease_probs_chunks.append(probs[:, eligible_arr].astype(np.float32))
        true_tokens_chunks.append(true_arr)
        sex_chunks.append(np.array(sex_batch, dtype=np.int32))
        age_band_chunks.append(np.array(age_band_batch, dtype=np.int8))

    # --- Aggregate ---
    mean_nll = nll_sum / n_points if n_points > 0 else float("nan")
    top_k_acc = {k: top_k_hits[k] / n_points for k in [1, 5, 20]}

    true_tokens_arr = np.concatenate(true_tokens_chunks)
    sex_arr = np.concatenate(sex_chunks)
    age_band_arr = np.concatenate(age_band_chunks)
    disease_probs_arr = (
        np.vstack(disease_probs_chunks) if disease_probs_chunks else np.empty((n_points, 0), dtype=np.float32)
    )

    # --- Per-disease AUROC ---
    per_disease_records: list[dict] = []
    for d_idx, d_tok in enumerate(eligible_list):
        labels = (true_tokens_arr == d_tok).astype(np.int32)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        try:
            auc = float(roc_auc_score(labels, disease_probs_arr[:, d_idx]))
        except ValueError:
            continue
        per_disease_records.append(
            {"disease": vocab[d_tok], "auc": auc, "n_positives": int(labels.sum())}
        )

    per_disease_df = (
        pd.DataFrame(per_disease_records).sort_values("auc", ascending=False)
        if per_disease_records
        else pd.DataFrame(columns=["disease", "auc", "n_positives"])
    )
    mean_auroc = float(per_disease_df["auc"].mean()) if len(per_disease_df) > 0 else float("nan")
    median_auroc = float(per_disease_df["auc"].median()) if len(per_disease_df) > 0 else float("nan")
    per_disease_df.to_csv(OUTPUTS_DIR / f"{name}_per_disease_auc.csv", index=False)

    # --- Stratified AUROC ---
    strata: list[tuple[str, np.ndarray]] = []
    if female_tok is not None:
        strata.append(("Female", sex_arr == female_tok))
    if male_tok is not None:
        strata.append(("Male", sex_arr == male_tok))
    strata += [
        ("age<40", age_band_arr == 0),
        ("age40-60", age_band_arr == 1),
        ("age>60", age_band_arr == 2),
    ]

    strat_records: list[dict] = []
    for stratum_name, mask in strata:
        if mask.sum() == 0:
            continue
        sub_true = true_tokens_arr[mask]
        sub_probs = disease_probs_arr[mask]
        for d_idx, d_tok in enumerate(eligible_list):
            sub_labels = (sub_true == d_tok).astype(np.int32)
            n_pos = int(sub_labels.sum())
            if n_pos == 0 or n_pos == len(sub_labels):
                continue
            try:
                auc = float(roc_auc_score(sub_labels, sub_probs[:, d_idx]))
            except ValueError:
                continue
            strat_records.append(
                {"disease": vocab[d_tok], "stratum": stratum_name, "auc": auc, "n_positives": n_pos}
            )

    strat_df = (
        pd.DataFrame(strat_records)
        if strat_records
        else pd.DataFrame(columns=["disease", "stratum", "auc", "n_positives"])
    )
    strat_df.to_csv(OUTPUTS_DIR / f"{name}_stratified_auc.csv", index=False)

    # --- Print summary ---
    n_dis = len(per_disease_df)
    print(f"\n=== {name} ===")
    print(f"{'n_points':<22} {n_points:,}")
    print(f"{'mean_nll (nats)':<22} {mean_nll:.4f}")
    print(f"{'top1_acc':<22} {top_k_acc[1]:.4f}")
    print(f"{'top5_acc':<22} {top_k_acc[5]:.4f}")
    print(f"{'top20_acc':<22} {top_k_acc[20]:.4f}")
    print(f"{'auroc_mean':<22} {mean_auroc:.4f}  (over {n_dis} diseases)")
    print(f"{'auroc_median':<22} {median_auroc:.4f}")

    return {
        "name": name,
        "mean_nll": mean_nll,
        "top1_acc": top_k_acc[1],
        "top5_acc": top_k_acc[5],
        "top20_acc": top_k_acc[20],
        "mean_auroc": mean_auroc,
        "median_auroc": median_auroc,
        "n_diseases_auroc": n_dis,
        "n_points": n_points,
    }
