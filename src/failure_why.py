"""Describe whether age or the previous event carries signal for each eligible disease.

Descriptive only: two measured quantities per disease, no cause is tested.

1. Age signal (val): AUROC of age at the last history event, used alone as the score
   for "the next event is this disease", over all val prediction points.
2. History signal (train minus patient 402867): lift of each previous token x,
       lift = P(next is d | previous is x) / P(next is d),
   over previous tokens seen at least 50 times.  The maximum lift and its token are
   reported, and again restricted to (x, d) pairs seen at least 5 times
   (max_lift_min5_train), because a maximum over many noisy ratios is inflated for
   rare diseases.

Both are computed for every eligible disease (rows of
outputs/transformer_per_disease_auc.csv), so the failure codes and the panel codes
can be compared with the whole distribution through the percentile columns.

Prerequisites: outputs of python -m src.baselines, src.run_eval and src.analysis.

Writes outputs/failure_why.csv and prints one line per failure code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.bootstrap import PANEL_CODES
from src.data import build_sequences, load_split, load_vocab

OUTPUTS_DIR = Path("outputs")
_EXCLUDED_TRAIN_PATIENT: int = 402867
_MIN_PREV: int = 50    # a previous token needs at least this many train occurrences
_MIN_JOINT: int = 5    # joint count threshold for max_lift_min5_train

Sequence = tuple[np.ndarray, np.ndarray]


def val_points(sequences: list[Sequence]) -> tuple[np.ndarray, np.ndarray]:
    """Prediction points of evaluate.py's protocol, without any model.

    At point k the history is positions 0..k-1 and the target is token k.

    Returns:
        age_last (n_points,) float64, age in years at the last history event (position k-1)
        true_tok (n_points,) int64, the true next token (position k)
    """
    age_chunks: list[np.ndarray] = []
    tok_chunks: list[np.ndarray] = []
    for tokens, ages in sequences:
        if len(tokens) < 2:
            continue
        age_chunks.append(ages[:-1].astype(np.float64) / 365.25)
        tok_chunks.append(tokens[1:].astype(np.int64))
    return np.concatenate(age_chunks), np.concatenate(tok_chunks)


def age_auroc(age_last: np.ndarray, true_tok: np.ndarray, tok: int) -> float:
    """AUROC of age alone for one disease; tied ages count as half (sklearn, as evaluate.py)."""
    return float(roc_auc_score((true_tok == tok).astype(np.int32), age_last))


def pair_counts(sequences: list[Sequence], vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Count (previous token, next token) pairs over all prediction points of sequences.

    Returns:
        pair       (vocab_size, vocab_size) int64; pair[x, d] = times d directly follows x
        prev_count (vocab_size,) int64; times x is the previous token
        next_count (vocab_size,) int64; times d is the next token
        total      number of pairs
    """
    pair = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    for tokens, _ in sequences:
        if len(tokens) < 2:
            continue
        np.add.at(pair, (tokens[:-1], tokens[1:]), 1)  # add.at counts repeated pairs correctly
    return pair, pair.sum(axis=1), pair.sum(axis=0), int(pair.sum())


def max_lift(
    pair: np.ndarray,
    prev_count: np.ndarray,
    next_count: np.ndarray,
    total: int,
    tok: int,
    min_joint: int,
) -> tuple[float, int | None, int, int]:
    """Largest lift for disease tok over the eligible previous tokens.

    A previous token x is a candidate if it occurs at least _MIN_PREV times and the
    pair (x, tok) occurs at least min_joint times.

    Returns:
        (lift, previous token, count of that previous token, joint count).
        (nan, None, 0, 0) if tok is never a next token or no candidate exists.
        The token is None when the largest lift is 0 (tok follows no candidate).
    """
    if next_count[tok] == 0:
        return float("nan"), None, 0, 0
    candidates = np.flatnonzero((prev_count >= _MIN_PREV) & (pair[:, tok] >= min_joint))
    if len(candidates) == 0:
        return float("nan"), None, 0, 0
    p_d = next_count[tok] / total                                  # P(next is d)
    p_d_given_x = pair[candidates, tok] / prev_count[candidates]   # P(next is d | previous is x)
    lift = p_d_given_x / p_d
    best = int(np.argmax(lift))  # ties: lowest token index
    x = int(candidates[best])
    if pair[x, tok] == 0:
        return 0.0, None, 0, 0
    return float(lift[best]), x, int(prev_count[x]), int(pair[x, tok])


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Share of the non-NaN entries at or below each value; NaN stays NaN."""
    valid = values[~np.isnan(values)]
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        if not np.isnan(v):
            out[i] = float((valid <= v).mean())
    return out


if __name__ == "__main__":
    train_df = load_split("train")
    val_df = load_split("val")
    vocab = load_vocab()
    train_df = train_df[train_df["person_id"] != _EXCLUDED_TRAIN_PATIENT]
    train_sequences = build_sequences(train_df)
    val_sequences = build_sequences(val_df)
    index_of = {name: i for i, name in enumerate(vocab)}

    # Eligible diseases and context columns, all read from committed files
    elig = pd.read_csv(OUTPUTS_DIR / "transformer_per_disease_auc.csv").rename(
        columns={"auc": "auc_transformer", "n_positives": "n_positives_val"}
    )
    for model in ("age_sex", "bigram"):
        other = pd.read_csv(OUTPUTS_DIR / f"{model}_per_disease_auc.csv")[["disease", "auc"]]
        elig = elig.merge(other.rename(columns={"auc": f"auc_{model}"}), on="disease", how="left", validate="one_to_one")
    failures = pd.read_csv(OUTPUTS_DIR / "failures.csv")[["disease", "failure_type"]]
    missing = set(failures["disease"]) - set(elig["disease"])
    if missing:
        raise RuntimeError(f"failure diseases not in the eligible list: {sorted(missing)}")

    age_last, true_tok = val_points(val_sequences)
    pair, prev_count, next_count, total = pair_counts(train_sequences, len(vocab))
    print(
        f"{len(elig)} eligible diseases; {len(age_last):,} val prediction points; {total:,} train pairs;"
        f" {int((prev_count >= _MIN_PREV).sum())} previous tokens with >= {_MIN_PREV} train occurrences"
    )

    rows: list[dict] = []
    for _, e in elig.iterrows():
        tok = index_of[e["disease"]]
        n_pos = int((true_tok == tok).sum())
        if n_pos != int(e["n_positives_val"]):
            raise RuntimeError(f"positives mismatch for {e['disease']}: {n_pos} vs {int(e['n_positives_val'])} in the CSV")
        auc_age = age_auroc(age_last, true_tok, tok)
        lift, x, n_prev, n_joint = max_lift(pair, prev_count, next_count, total, tok, min_joint=0)
        lift5, x5, n_prev5, n_joint5 = max_lift(pair, prev_count, next_count, total, tok, min_joint=_MIN_JOINT)
        rows.append({
            "disease": e["disease"],
            "code": e["disease"][:3],
            "age_auroc_val": auc_age,
            "age_distance_val": abs(auc_age - 0.5),
            "max_lift_train": lift,
            "max_lift_prev_token": vocab[x] if x is not None else "",
            "max_lift_n_prev": n_prev,
            "max_lift_n_joint": n_joint,
            "max_lift_min5_train": lift5,
            "max_lift_min5_prev_token": vocab[x5] if x5 is not None else "",
            "max_lift_min5_n_prev": n_prev5,
            "max_lift_min5_n_joint": n_joint5,
            "n_target_train": int(next_count[tok]),
            "n_positives_val": n_pos,
            "auc_transformer": float(e["auc_transformer"]),
            "auc_age_sex": float(e["auc_age_sex"]),
            "auc_bigram": float(e["auc_bigram"]),
        })

    df = pd.DataFrame(rows)
    # Percentiles within the eligible diseases: share of eligible diseases at or below the value
    df["age_distance_pct"] = percentile_rank(df["age_distance_val"].to_numpy())
    df["max_lift_pct"] = percentile_rank(df["max_lift_train"].to_numpy())
    df["max_lift_min5_pct"] = percentile_rank(df["max_lift_min5_train"].to_numpy())

    df = df.merge(failures, on="disease", how="left", validate="one_to_one")
    df["in_failures"] = df["failure_type"].notna()
    df["failure_type"] = df["failure_type"].fillna("")
    df["in_panel"] = df["code"].isin(PANEL_CODES)

    df = df[[
        "disease", "code", "in_failures", "failure_type", "in_panel",
        "age_auroc_val", "age_distance_val", "age_distance_pct",
        "max_lift_train", "max_lift_prev_token", "max_lift_n_prev", "max_lift_n_joint", "max_lift_pct",
        "max_lift_min5_train", "max_lift_min5_prev_token", "max_lift_min5_n_prev", "max_lift_min5_n_joint",
        "max_lift_min5_pct",
        "n_target_train", "n_positives_val", "auc_transformer", "auc_age_sex", "auc_bigram",
    ]]
    df.to_csv(OUTPUTS_DIR / "failure_why.csv", index=False)

    # Panel codes without a row are named, never dropped silently
    for code in PANEL_CODES:
        if code not in set(df["code"]):
            print(f"panel code {code}: not eligible, no row")

    # One line per failure code, in failures.csv order; numbers only
    print("\nfailure codes (pct = share of eligible diseases at or below the value):")
    by_disease = df.set_index("disease")
    for disease in failures["disease"]:
        r = by_disease.loc[disease]
        print(
            f"{r['code']}"
            f" | age_auroc_val {r['age_auroc_val']:.4f} distance {r['age_distance_val']:.4f} pct {r['age_distance_pct']:.3f}"
            f" | max_lift_train {r['max_lift_train']:.1f} prev '{r['max_lift_prev_token'].split(' ')[0]}'"
            f" n_prev {r['max_lift_n_prev']} n_joint {r['max_lift_n_joint']} pct {r['max_lift_pct']:.3f}"
            f" | max_lift_min5_train {r['max_lift_min5_train']:.1f} prev '{r['max_lift_min5_prev_token'].split(' ')[0]}'"
            f" n_joint {r['max_lift_min5_n_joint']} pct {r['max_lift_min5_pct']:.3f}"
            f" | n_target_train {r['n_target_train']} n_positives_val {r['n_positives_val']}"
            f" | auc transformer {r['auc_transformer']:.4f} age_sex {r['auc_age_sex']:.4f} bigram {r['auc_bigram']:.4f}"
        )
    print(f"\nSaved {OUTPUTS_DIR / 'failure_why.csv'}")
