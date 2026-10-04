"""Do the trained token embeddings group diagnoses by ICD-10 chapter?

Tests, on this model, the finding reported for Delphi-2M (Shmatko et al., Nature 2025)
that diagnosis embeddings cluster by ICD-10 chapter.  The method was fixed before running:

1. Embeddings: token_emb.weight from checkpoints/best.pt, on CPU.
2. Tokens: diagnosis codes with at least 20 events in train minus patient 402867.
   Code = first 3 characters of the label.
3. Chapters: WHO ICD-10 chapter ranges (CHAPTERS below).
4. Mean-centre the selected embeddings, then cosine similarity.
5. Statistic: mean within-chapter cosine minus mean between-chapter cosine.
6. Test 1: shuffle chapter labels 1000 times, seed 0; one-sided p-value.
7. Test 2: frequency control; shuffle labels only within deciles of train frequency.
8. Test 3: negative control; the same statistic and the same label shuffle on an
   untrained HealthTransformer() initialised with torch.manual_seed(42).
9. Declared check: the cosine rank of H36 and G63 among the neighbours of E11.
10. Nearest neighbours: the 3 nearest codes for the 10 most frequent codes in train.

Writes to outputs/:
    embedding_chapters.json
    embedding_neighbours.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import load_split, load_vocab
from src.model import HealthTransformer

OUTPUTS_DIR = Path("outputs")
CHECKPOINT_PATH = Path("checkpoints/best.pt")
_EXCLUDED_TRAIN_PATIENT: int = 402867
_MIN_COUNT: int = 20          # a code needs at least this many train events to be tested
_N_PERM: int = 1000           # label shuffles per test
_PERM_SEED: int = 0           # seed of the generator for every label-shuffle test
_UNTRAINED_SEED: int = 42     # torch seed for the untrained negative control
_N_DECILES: int = 10
_ANCHOR: str = "E11"                          # declared check: neighbours of this code
_TARGETS: tuple[str, ...] = ("H36", "G63")    # declared check: codes whose rank is reported
_N_FREQUENT: int = 10         # nearest neighbours are listed for this many most frequent codes
_N_NEIGHBOURS: int = 3
_NOT_TESTABLE: str = "not testable"

# WHO ICD-10 chapters: (numeral, first code, last code, title)
CHAPTERS: list[tuple[str, str, str, str]] = [
    ("I", "A00", "B99", "Certain infectious and parasitic diseases"),
    ("II", "C00", "D48", "Neoplasms"),
    ("III", "D50", "D89", "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"),
    ("IV", "E00", "E90", "Endocrine, nutritional and metabolic diseases"),
    ("V", "F00", "F99", "Mental and behavioural disorders"),
    ("VI", "G00", "G99", "Diseases of the nervous system"),
    ("VII", "H00", "H59", "Diseases of the eye and adnexa"),
    ("VIII", "H60", "H95", "Diseases of the ear and mastoid process"),
    ("IX", "I00", "I99", "Diseases of the circulatory system"),
    ("X", "J00", "J99", "Diseases of the respiratory system"),
    ("XI", "K00", "K93", "Diseases of the digestive system"),
    ("XII", "L00", "L99", "Diseases of the skin and subcutaneous tissue"),
    ("XIII", "M00", "M99", "Diseases of the musculoskeletal system and connective tissue"),
    ("XIV", "N00", "N99", "Diseases of the genitourinary system"),
    ("XV", "O00", "O99", "Pregnancy, childbirth and the puerperium"),
    ("XVI", "P00", "P96", "Certain conditions originating in the perinatal period"),
    ("XVII", "Q00", "Q99", "Congenital malformations, deformations and chromosomal abnormalities"),
    ("XVIII", "R00", "R99", "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"),
    ("XIX", "S00", "T98", "Injury, poisoning and certain other consequences of external causes"),
    ("XX", "V01", "Y98", "External causes of morbidity and mortality"),
    ("XXI", "Z00", "Z99", "Factors influencing health status and contact with health services"),
    ("XXII", "U00", "U85", "Codes for special purposes"),
]

_CODE_PATTERN = re.compile(r"[A-Z][0-9]{2}")


# ---------------------------------------------------------------------------
# Tokens and chapters
# ---------------------------------------------------------------------------

def chapter_of(code: str) -> str:
    """Return the chapter numeral of a 3-character ICD-10 code.

    A letter followed by two digits compares correctly as a string, so
    "A00" <= code <= "B99" is the range test.  Raises if no range matches,
    so no code is ever assigned to a chapter silently.
    """
    for numeral, first, last, _ in CHAPTERS:
        if first <= code <= last:
            return numeral
    raise ValueError(f"code {code} falls in no WHO ICD-10 chapter range")


def select_tokens(train_df: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    """Select the diagnosis codes to test.

    A label is a diagnosis code if its first 3 characters are a capital letter and two
    digits; this drops Padding, Female and Male.  A code is kept if it occurs at least
    _MIN_COUNT times in train_df.

    Args:
        train_df: train events, patient 402867 already removed.
        vocab:    output of load_vocab().

    Returns:
        DataFrame, one row per selected token in ascending token index, with columns
        token, code, label, chapter, train_count.
    """
    counts = np.bincount(train_df["token"].to_numpy(), minlength=len(vocab))
    rows: list[dict] = []
    for tok, label in enumerate(vocab):
        code = label[:3]
        if _CODE_PATTERN.fullmatch(code) is None:
            continue
        if counts[tok] < _MIN_COUNT:
            continue
        rows.append({
            "token": tok, "code": code, "label": label,
            "chapter": chapter_of(code), "train_count": int(counts[tok]),
        })
    sel = pd.DataFrame(rows)
    if sel["code"].duplicated().any():
        raise RuntimeError("two vocabulary entries share the same 3-character code")
    return sel


# ---------------------------------------------------------------------------
# Similarity and the statistic
# ---------------------------------------------------------------------------

def centred_cosine(emb: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows after subtracting the mean row.

    Args:
        emb: (n_tokens, d_model) embedding rows of the selected tokens.

    Returns:
        (n_tokens, n_tokens) float64 cosine matrix.
    """
    x = emb.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x @ x.T


def chapter_statistic(pair_cos: np.ndarray, same_chapter: np.ndarray) -> tuple[float, float, float]:
    """Mean within-chapter cosine minus mean between-chapter cosine.

    Args:
        pair_cos:     cosine of every unordered pair of tokens, each pair once.
        same_chapter: boolean, True where the pair's two tokens share a chapter.

    Returns:
        (statistic, mean within, mean between).
    """
    within = float(pair_cos[same_chapter].mean())
    between = float(pair_cos[~same_chapter].mean())
    return within - between, within, between


def permutation_p(
    cos: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    n_perm: int = _N_PERM,
    seed: int = _PERM_SEED,
) -> dict:
    """Label-shuffle test of the chapter statistic, one-sided.

    Args:
        cos:    (n, n) cosine matrix.
        labels: (n,) integer chapter id per token.
        groups: None to shuffle labels freely; otherwise (n,) group id per token, and
                labels are shuffled only among tokens of the same group.
        n_perm: number of shuffles.
        seed:   seed of the numpy generator.

    Returns:
        Dict with statistic, mean_within, mean_between, p_value, null_mean, null_max,
        n_perm.  p_value = (1 + number of shuffles with statistic >= observed) / (n_perm + 1).
    """
    row, col = np.triu_indices(len(labels), k=1)  # every unordered pair once
    pair_cos = cos[row, col]
    observed, within, between = chapter_statistic(pair_cos, labels[row] == labels[col])

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        if groups is None:
            shuffled = rng.permutation(labels)
        else:
            shuffled = labels.copy()
            for g in np.unique(groups):
                idx = np.flatnonzero(groups == g)
                shuffled[idx] = rng.permutation(labels[idx])
        null[b], _, _ = chapter_statistic(pair_cos, shuffled[row] == shuffled[col])

    return {
        "statistic": observed,
        "mean_within": within,
        "mean_between": between,
        "p_value": float((1 + int((null >= observed).sum())) / (n_perm + 1)),
        "null_mean": float(null.mean()),
        "null_max": float(null.max()),
        "n_perm": int(n_perm),
    }


def frequency_deciles(counts: np.ndarray) -> np.ndarray:
    """Decile of train frequency per token, by rank, so the ten groups have equal size.

    Ties in count are broken by position (ascending token index).  Ranks are used rather
    than count thresholds because many rare codes share the same count.
    """
    n = len(counts)
    order = np.argsort(counts, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank * _N_DECILES // n


def within_chapter_means(cos: np.ndarray, sel: pd.DataFrame) -> dict[str, dict]:
    """Mean within-chapter cosine for each chapter; description only, not a test."""
    out: dict[str, dict] = {}
    chapters = sel["chapter"].to_numpy()
    for numeral, _, _, title in CHAPTERS:
        idx = np.flatnonzero(chapters == numeral)
        if len(idx) == 0:
            continue
        mean_cos = None  # a chapter with one selected code has no within-chapter pair
        if len(idx) >= 2:
            r, c = np.triu_indices(len(idx), k=1)
            mean_cos = float(cos[np.ix_(idx, idx)][r, c].mean())
        out[numeral] = {"title": title, "n_tokens": int(len(idx)), "mean_within_cosine": mean_cos}
    return out


# ---------------------------------------------------------------------------
# Declared check and nearest neighbours
# ---------------------------------------------------------------------------

def declared_check(cos: np.ndarray, sel: pd.DataFrame) -> dict:
    """Cosine rank of each target code among the neighbours of the anchor code.

    Rank 1 is the nearest code to the anchor; the anchor itself is not a candidate.
    A code that failed the count threshold is reported as "not testable".
    """
    codes = sel["code"].tolist()
    out: dict = {"anchor": _ANCHOR, "n_candidates": len(codes) - 1, "targets": {}}
    if _ANCHOR not in codes:
        out["anchor_status"] = _NOT_TESTABLE
        for target in _TARGETS:
            out["targets"][target] = {"status": _NOT_TESTABLE}
        return out
    out["anchor_status"] = "testable"
    a = codes.index(_ANCHOR)
    sims = cos[a].copy()
    sims[a] = -np.inf  # the anchor is not its own neighbour
    for target in _TARGETS:
        if target not in codes:
            out["targets"][target] = {"status": _NOT_TESTABLE}
            continue
        t = codes.index(target)
        out["targets"][target] = {
            "status": "testable",
            "rank": 1 + int((sims > sims[t]).sum()),
            "cosine": float(sims[t]),
        }
    return out


def nearest_neighbours(cos: np.ndarray, sel: pd.DataFrame) -> pd.DataFrame:
    """The _N_NEIGHBOURS nearest codes of each of the _N_FREQUENT most frequent codes."""
    counts = sel["train_count"].to_numpy()
    frequent = np.argsort(-counts, kind="stable")[:_N_FREQUENT]
    rows: list[dict] = []
    for i in frequent:
        sims = cos[i].copy()
        sims[i] = -np.inf  # a code is not its own neighbour
        for rank, j in enumerate(np.argsort(-sims, kind="stable")[:_N_NEIGHBOURS], start=1):
            rows.append({
                "code": sel["code"].iloc[i],
                "label": sel["label"].iloc[i],
                "chapter": sel["chapter"].iloc[i],
                "train_count": int(counts[i]),
                "neighbour_rank": rank,
                "neighbour_code": sel["code"].iloc[j],
                "neighbour_label": sel["label"].iloc[j],
                "neighbour_chapter": sel["chapter"].iloc[j],
                "cosine": float(sims[j]),
                "same_chapter": bool(sel["chapter"].iloc[i] == sel["chapter"].iloc[j]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(exist_ok=True)

    train_df = load_split("train")
    vocab = load_vocab()
    train_df = train_df[train_df["person_id"] != _EXCLUDED_TRAIN_PATIENT]

    sel = select_tokens(train_df, vocab)
    tokens = sel["token"].to_numpy()
    chapter_names = sorted(sel["chapter"].unique(), key=[c[0] for c in CHAPTERS].index)
    labels = sel["chapter"].map({name: i for i, name in enumerate(chapter_names)}).to_numpy()
    print(f"selected {len(sel)} diagnosis codes with >= {_MIN_COUNT} train events, in {len(chapter_names)} chapters")

    # Trained embeddings, read straight from the checkpoint on CPU
    state = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"), weights_only=True)
    cos = centred_cosine(state["token_emb.weight"].numpy()[tokens])

    # Test 1: free label shuffle.  Test 2: shuffle within deciles of train frequency.
    test1 = permutation_p(cos, labels, groups=None)
    deciles = frequency_deciles(sel["train_count"].to_numpy())
    test2 = permutation_p(cos, labels, groups=deciles)

    # Test 3: untrained model, same tokens, labels, centring, statistic and label shuffle
    torch.manual_seed(_UNTRAINED_SEED)
    untrained = HealthTransformer()
    cos_untrained = centred_cosine(untrained.token_emb.weight.detach().numpy()[tokens])
    negative = permutation_p(cos_untrained, labels, groups=None)

    check = declared_check(cos, sel)
    neighbours = nearest_neighbours(cos, sel)

    summary = {
        "checkpoint": str(CHECKPOINT_PATH),
        "embedding": "token_emb.weight (input embedding), mean-centred over the selected tokens, cosine",
        "min_train_count": _MIN_COUNT,
        "n_tokens": int(len(sel)),
        "n_chapters": int(len(chapter_names)),
        "n_perm": _N_PERM,
        "perm_seed": _PERM_SEED,
        "p_value_rule": "one-sided: (1 + shuffles with statistic >= observed) / (n_perm + 1)",
        "statistic": test1["statistic"],
        "mean_within": test1["mean_within"],
        "mean_between": test1["mean_between"],
        "test1_label_shuffle": {k: test1[k] for k in ("p_value", "null_mean", "null_max")},
        "test2_frequency_control": {
            "grouping": f"{_N_DECILES} equal-size groups by rank of train count",
            **{k: test2[k] for k in ("p_value", "null_mean", "null_max")},
        },
        "test3_negative_control": {
            "model": f"untrained HealthTransformer(), torch.manual_seed({_UNTRAINED_SEED})",
            **{k: negative[k] for k in ("statistic", "mean_within", "mean_between", "p_value", "null_mean", "null_max")},
        },
        "declared_check": check,
        "per_chapter_description_only": within_chapter_means(cos, sel),
    }

    with open(OUTPUTS_DIR / "embedding_chapters.json", "w") as f:
        json.dump(summary, f, indent=2)
    neighbours.to_csv(OUTPUTS_DIR / "embedding_neighbours.csv", index=False)

    print(f"statistic (within - between): {test1['statistic']:.4f}  (within {test1['mean_within']:.4f}, between {test1['mean_between']:.4f})")
    print(f"test 1 label shuffle:       p = {test1['p_value']:.4f}  (null mean {test1['null_mean']:.4f}, null max {test1['null_max']:.4f})")
    print(f"test 2 frequency control:   p = {test2['p_value']:.4f}  (null mean {test2['null_mean']:.4f}, null max {test2['null_max']:.4f})")
    print(f"test 3 untrained model:     statistic {negative['statistic']:.4f}, p = {negative['p_value']:.4f}")
    print(f"smallest possible p with {_N_PERM} shuffles: {1 / (_N_PERM + 1):.4f}")
    for target, res in check["targets"].items():
        if res["status"] == _NOT_TESTABLE:
            print(f"declared check {_ANCHOR} -> {target}: {_NOT_TESTABLE}")
        else:
            print(f"declared check {_ANCHOR} -> {target}: rank {res['rank']} of {check['n_candidates']} (cosine {res['cosine']:.4f})")
    print(f"Saved {OUTPUTS_DIR / 'embedding_chapters.json'}")
    print(f"Saved {OUTPUTS_DIR / 'embedding_neighbours.csv'}")
