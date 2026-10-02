"""Baseline predict_fn factories for health trajectory evaluation.

All baselines are fitted on train sequences only. Patient 402867 is excluded
from every training computation: it has no sex token as its first event and
therefore fails the sex-token sanity check, making it anomalous training data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data import build_sequences, load_split, load_vocab, split_train_dev
from src.evaluate import evaluate

OUTPUTS_DIR = Path("outputs")

_EXCLUDED_TRAIN_PATIENT: int = 402867  # no sex token as first event
_ALPHA: float = 0.01                   # add-alpha for marginal smoothing
_K: float = 20.0                       # interpolation constant: λ = n / (n + K)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _valid_mask(vocab_size: int, sex_indices: set[int]) -> np.ndarray:
    """Boolean mask of tokens that can appear as prediction targets.

    Excludes index 0 (padding, never in sequences) and sex tokens
    (always the first event, never a prediction target).
    """
    mask = np.ones(vocab_size, dtype=bool)
    mask[0] = False
    for s in sex_indices:
        mask[s] = False
    return mask


def _fit_marginal_probs(
    train_sequences: list[tuple[np.ndarray, np.ndarray]],
    vocab_size: int,
    sex_indices: set[int],
) -> np.ndarray:
    """Return smoothed marginal distribution over vocab from train target counts."""
    counts = np.zeros(vocab_size, dtype=np.float64)
    for tokens, _ in train_sequences:
        for tok in tokens[1:]:
            counts[int(tok)] += 1
    mask = _valid_mask(vocab_size, sex_indices)
    counts[~mask] = 0.0
    counts[mask] += _ALPHA
    return (counts / counts.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Baseline factories
# ---------------------------------------------------------------------------

def fit_marginal(
    train_sequences: list[tuple[np.ndarray, np.ndarray]],
    vocab: list[str],
    sex_indices: set[int],
) -> Callable[[list[tuple[np.ndarray, np.ndarray]]], np.ndarray]:
    """Fit a marginal next-token baseline on train sequences.

    Predicted distribution: overall train target token frequencies with
    add-alpha 0.01 smoothing. The same distribution is returned for every
    prefix regardless of context.
    """
    vocab_size = len(vocab)
    probs = _fit_marginal_probs(train_sequences, vocab_size, sex_indices)

    def predict(prefixes: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        return np.tile(probs, (len(prefixes), 1))

    return predict


def fit_age_sex(
    train_sequences: list[tuple[np.ndarray, np.ndarray]],
    vocab: list[str],
    sex_indices: set[int],
    K: float = _K,
) -> Callable[[list[tuple[np.ndarray, np.ndarray]]], np.ndarray]:
    """Fit an age-sex conditional baseline on train sequences.

    Conditioning: patient sex (tokens[0]) × 5-year age band of the current
    event (floor(ages[-1] in years / 5), capped at band 20 for age ≥ 100).

    Predicted distribution uses interpolated backoff:
        p = λ · p_conditional + (1 − λ) · p_marginal
        λ = n / (n + K)
    where n is the total event count in the conditioning cell and K is the
    interpolation constant (default 20, tunable via tune_K).

    Reason for interpolation rather than hard backoff with near-zero alpha:
    a near-zero alpha makes conditional baselines overconfident and artificially
    weak on cross-entropy, which would make them easy to beat and uninformative.
    """
    vocab_size = len(vocab)
    n_age_bands = 21  # bands 0..19 cover 0–99 years in 5-year steps; band 20 = 100+

    female_tok = next((i for i, v in enumerate(vocab) if v == "Female"), None)
    male_tok = next((i for i, v in enumerate(vocab) if v == "Male"), None)

    def _sex_to_idx(tok: int) -> int | None:
        if tok == female_tok:
            return 0
        if tok == male_tok:
            return 1
        return None

    def _age_band(age_days: float) -> int:
        return min(int(age_days / 365.25 / 5), 20)

    mask = _valid_mask(vocab_size, sex_indices)
    counts = np.zeros((2, n_age_bands, vocab_size), dtype=np.float64)
    cell_totals = np.zeros((2, n_age_bands), dtype=np.float64)

    for tokens, ages in train_sequences:
        s_idx = _sex_to_idx(int(tokens[0]))
        if s_idx is None:
            continue
        for k in range(1, len(tokens)):
            ab = _age_band(float(ages[k - 1]))
            tok = int(tokens[k])
            if mask[tok]:
                counts[s_idx, ab, tok] += 1.0
                cell_totals[s_idx, ab] += 1.0

    # Normalised conditional distributions (no smoothing here; marginal handles it)
    cond_probs = np.zeros((2, n_age_bands, vocab_size), dtype=np.float32)
    for s in range(2):
        for ab in range(n_age_bands):
            n = cell_totals[s, ab]
            if n > 0:
                p = counts[s, ab].copy()
                p[~mask] = 0.0
                cond_probs[s, ab] = (p / p.sum()).astype(np.float32)

    p_marginal = _fit_marginal_probs(train_sequences, vocab_size, sex_indices)

    def predict(prefixes: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        n_pref = len(prefixes)
        s_idx_arr = np.zeros(n_pref, dtype=np.int32)
        ab_arr = np.zeros(n_pref, dtype=np.int32)
        valid = np.ones(n_pref, dtype=bool)

        for i, (tokens, ages) in enumerate(prefixes):
            s = _sex_to_idx(int(tokens[0]))
            if s is None:
                valid[i] = False
            else:
                s_idx_arr[i] = s
            ab_arr[i] = _age_band(float(ages[-1]))

        n_cell = cell_totals[s_idx_arr, ab_arr]          # (n_pref,)
        lam = (n_cell / (n_cell + K)).astype(np.float32)
        out = lam[:, None] * cond_probs[s_idx_arr, ab_arr] + (1.0 - lam[:, None]) * p_marginal
        out[~valid] = p_marginal
        return out

    return predict


def fit_bigram(
    train_sequences: list[tuple[np.ndarray, np.ndarray]],
    vocab: list[str],
    sex_indices: set[int],
    K: float = _K,
) -> Callable[[list[tuple[np.ndarray, np.ndarray]]], np.ndarray]:
    """Fit a bigram (previous-token) conditional baseline on train sequences.

    Conditioning: the most recent token in the prefix, tokens[-1].

    Predicted distribution uses interpolated backoff:
        p = λ · p_conditional + (1 − λ) · p_marginal
        λ = n / (n + K)
    where n is the total count of training observations following that token
    and K is the interpolation constant (default 20, tunable via tune_K).

    Reason for interpolation rather than hard backoff with near-zero alpha:
    a near-zero alpha makes conditional baselines overconfident and artificially
    weak on cross-entropy, which would make them easy to beat and uninformative.
    """
    vocab_size = len(vocab)
    mask = _valid_mask(vocab_size, sex_indices)

    counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    token_totals = np.zeros(vocab_size, dtype=np.float32)

    for tokens, _ in train_sequences:
        for k in range(1, len(tokens)):
            prev = int(tokens[k - 1])
            tgt = int(tokens[k])
            if mask[tgt]:
                counts[prev, tgt] += 1.0
                token_totals[prev] += 1.0

    # Normalised conditional distributions
    cond_probs = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for prev in range(vocab_size):
        n = float(token_totals[prev])
        if n > 0:
            p = counts[prev].copy()
            p[~mask] = 0.0
            cond_probs[prev] = p / p.sum()

    p_marginal = _fit_marginal_probs(train_sequences, vocab_size, sex_indices)

    def predict(prefixes: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        prev_tokens = np.array([int(t[-1]) for t, _ in prefixes], dtype=np.int64)
        n_cell = token_totals[prev_tokens].astype(np.float32)
        lam = n_cell / (n_cell + K)
        return lam[:, None] * cond_probs[prev_tokens] + (1.0 - lam[:, None]) * p_marginal

    return predict


# ---------------------------------------------------------------------------
# Tuning helpers
# ---------------------------------------------------------------------------

def _mean_nll(
    predict_fn: Callable[[list[tuple[np.ndarray, np.ndarray]]], np.ndarray],
    sequences: list[tuple[np.ndarray, np.ndarray]],
    batch_size: int = 256,
) -> float:
    """Mean NLL over all prediction points in sequences (no AUROC, no I/O)."""
    nll_sum = 0.0
    n_points = 0
    for batch_start in range(0, len(sequences), batch_size):
        batch = sequences[batch_start : batch_start + batch_size]
        prefixes: list[tuple[np.ndarray, np.ndarray]] = []
        true_toks: list[int] = []
        for tokens, ages in batch:
            if len(tokens) < 2:
                continue
            for k in range(1, len(tokens)):
                prefixes.append((tokens[:k], ages[:k]))
                true_toks.append(int(tokens[k]))
        if not prefixes:
            continue
        probs = predict_fn(prefixes)
        true_arr = np.array(true_toks, dtype=np.int64)
        n = len(prefixes)
        true_probs = np.clip(probs[np.arange(n), true_arr], 1e-12, None)
        nll_sum += float(-np.log(true_probs).sum())
        n_points += n
    return nll_sum / n_points if n_points > 0 else float("nan")


def tune_K(
    fitter: Callable,
    name: str,
    train_part: list[tuple[np.ndarray, np.ndarray]],
    dev_part: list[tuple[np.ndarray, np.ndarray]],
    vocab: list[str],
    sex_indices: set[int],
    Ks: tuple[int, ...] = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000),
) -> tuple[int, list[dict]]:
    """Grid-search K for a conditional baseline using dev_part NLL.

    Fits fitter(train_part, vocab, sex_indices, K=k) for each k in Ks,
    scores on dev_part mean NLL, prints a table, returns (best_K, records).
    Records have columns: baseline, K, dev_nll.
    Val is never touched; dev_part comes from split_train_dev(train_sequences).
    """
    records: list[dict] = []
    print(f"\n=== Tuning K for {name} ===")
    print(f"{'K':>6}  dev_nll")
    print("-" * 18)
    for k in Ks:
        fn = fitter(train_part, vocab, sex_indices, K=float(k))
        nll = _mean_nll(fn, dev_part)
        records.append({"baseline": name, "K": k, "dev_nll": nll})
        print(f"{k:>6}  {nll:.4f}")
    best = min(records, key=lambda r: r["dev_nll"])
    best_K = int(best["K"])
    print(f"Best K = {best_K}")
    return best_K, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_df = load_split("train")
    val_df = load_split("val")
    vocab = load_vocab()

    # Drop patient 402867: no sex token as first event (fails sex-token sanity check).
    train_df = train_df[train_df["person_id"] != _EXCLUDED_TRAIN_PATIENT]

    train_sequences = build_sequences(train_df)
    val_sequences = build_sequences(val_df)

    sex_indices = {i for i, v in enumerate(vocab) if v in ("Female", "Male")}
    unseen: set[int] = set(val_df["token"]) - set(train_df["token"])

    # Split train into train_part / dev_part for K tuning.
    # Val is never used for tuning.
    train_part, dev_part = split_train_dev(train_sequences)

    # Tune K for conditional baselines on dev_part NLL.
    best_K_age_sex, records_age_sex = tune_K(
        fit_age_sex, "age_sex", train_part, dev_part, vocab, sex_indices
    )
    best_K_bigram, records_bigram = tune_K(
        fit_bigram, "bigram", train_part, dev_part, vocab, sex_indices
    )

    OUTPUTS_DIR.mkdir(exist_ok=True)
    tuning_df = pd.DataFrame(records_age_sex + records_bigram)
    tuning_df.to_csv(OUTPUTS_DIR / "baseline_K_tuning.csv", index=False)
    print(f"\nSaved {OUTPUTS_DIR}/baseline_K_tuning.csv")
    print(f"Chosen K: age_sex={best_K_age_sex}  bigram={best_K_bigram}")

    # Refit on full train (minus 402867) with tuned K, then evaluate on val.
    marginal_fn = fit_marginal(train_sequences, vocab, sex_indices)
    age_sex_fn = fit_age_sex(train_sequences, vocab, sex_indices, K=float(best_K_age_sex))
    bigram_fn = fit_bigram(train_sequences, vocab, sex_indices, K=float(best_K_bigram))

    results = []
    for bname, fn in [
        ("marginal", marginal_fn),
        ("age_sex", age_sex_fn),
        ("bigram", bigram_fn),
    ]:
        r = evaluate(fn, val_sequences, vocab, bname, excluded_from_auroc=unseen)
        results.append(r)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUTS_DIR / "baselines_summary.csv", index=False)
    print(f"\nSaved {OUTPUTS_DIR}/baselines_summary.csv")
