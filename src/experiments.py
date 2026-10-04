"""Evaluate the extra training runs: two more seeds and two training fractions.

Prerequisites: outputs of python -m src.baselines, src.run_eval and src.analysis, plus
the four tagged runs of src.train:

    python -m src.train --seed 1 --tag seed1
    python -m src.train --seed 2 --tag seed2
    python -m src.train --train-frac 0.25 --tag frac025
    python -m src.train --train-frac 0.5 --tag frac050

The seed 42, fraction 1.0 transformer is the committed run; it is read from
outputs/model_comparison.csv and never re-evaluated.  The tags are trusted: a
checkpoint does not record the seed or fraction it was trained with.

Writes to outputs/:
    transformer_{seed1,seed2,frac025,frac050}_{per_disease,stratified}_auc.csv  via evaluate()
    age_sex_{frac025,frac050,frac100}_{per_disease,stratified}_auc.csv          via evaluate()
    seeds_summary.csv      one row per seed (42, 1, 2), then mean, min, max
    data_efficiency.csv    one row per fraction per model
    data_efficiency.png    mean AUROC against training fraction, both models
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.baselines import fit_age_sex
from src.data import build_sequences, load_split, load_vocab, split_train_dev
from src.evaluate import evaluate
from src.model import HealthTransformer, make_predict_fn, select_device

OUTPUTS_DIR = Path("outputs")
CHECKPOINT_DIR = Path("checkpoints")
_EXCLUDED_TRAIN_PATIENT: int = 402867
_COMMITTED_SEED: int = 42                                  # seed of the committed transformer run
_SEED_TAGS: dict[int, str] = {1: "seed1", 2: "seed2"}      # extra seeds -> train.py tag
_FRAC_TAGS: dict[float, str] = {0.25: "frac025", 0.5: "frac050"}  # fractions -> train.py tag
# Subset sizes that train.py printed for the frac025 and frac050 runs
_EXPECTED_SUBSET_SIZE: dict[float, int] = {0.25: 1607, 0.5: 3214}

Sequence = tuple[np.ndarray, np.ndarray]
PredictFn = Callable[[list[Sequence]], np.ndarray]


# ---------------------------------------------------------------------------
# Training subsets and their checks
# ---------------------------------------------------------------------------

def nested_subset(train_part: list[Sequence], frac: float) -> list[Sequence]:
    """Return the training subset train.py uses for --train-frac frac.

    Mirrors the rule in src/train.py, which sits inside its __main__ block and
    cannot be imported: one shuffle of train_part with numpy generator seed 0,
    keep the first round(frac * n) patients.  frac = 1 returns train_part
    unshuffled.  Subsets are nested: a smaller fraction is a prefix of a larger one.
    """
    if frac >= 1.0:
        return train_part
    order = np.random.default_rng(0).permutation(len(train_part))
    n_keep = round(frac * len(train_part))
    return [train_part[i] for i in order[:n_keep]]


def patient_ids(sequences: list[Sequence], id_of: dict[int, int]) -> set[int]:
    """Person ids of a list of sequences.

    Args:
        sequences: sequences taken (not copied) from the list build_sequences returned.
        id_of:     lookup from id(tokens array) to person_id.
    """
    ids = {id_of[id(tokens)] for tokens, _ in sequences}
    if len(ids) != len(sequences):
        raise RuntimeError("patient_ids: a patient appears more than once in a subset")
    return ids


def check_splits(
    train_sequences: list[Sequence],
    id_of: dict[int, int],
) -> tuple[list[Sequence], dict[float, list[Sequence]]]:
    """Assert the split and subset properties; return train_part and the subsets by fraction.

    Checks, each printed:
    1. the dev patients are identical for all five run configurations and share no
       patient with that run's training subset;
    2. the 0.25 subset is contained in the 0.5 subset;
    3. the subset sizes equal the counts train.py printed.

    Check 1 verifies the rule: split_train_dev uses its own fixed seed, so --seed
    cannot change it.  It cannot inspect what the past training runs actually did.
    """
    runs: list[tuple[str, float]] = (
        [(f"seed{_COMMITTED_SEED}", 1.0)]
        + [(tag, 1.0) for tag in _SEED_TAGS.values()]
        + [(tag, frac) for frac, tag in _FRAC_TAGS.items()]
    )
    dev_ids_ref: set[int] | None = None
    for run_name, frac in runs:
        train_part, dev_part = split_train_dev(train_sequences)  # same call every run makes
        dev_ids = patient_ids(dev_part, id_of)
        train_ids = patient_ids(nested_subset(train_part, frac), id_of)
        if dev_ids_ref is None:
            dev_ids_ref = dev_ids
        assert dev_ids == dev_ids_ref, f"dev patients differ for run {run_name}"
        assert not (dev_ids & train_ids), f"run {run_name} trains on dev patients"
    print(
        f"CHECK dev split: identical {len(dev_ids_ref):,} dev patients for all {len(runs)} runs"
        f" ({', '.join(name for name, _ in runs)}); no dev patient in any training subset  OK"
    )

    train_part, _ = split_train_dev(train_sequences)
    subsets = {frac: nested_subset(train_part, frac) for frac in (0.25, 0.5, 1.0)}
    ids_025 = patient_ids(subsets[0.25], id_of)
    ids_050 = patient_ids(subsets[0.5], id_of)
    assert ids_025 <= ids_050, "the 0.25 subset is not contained in the 0.5 subset"
    print(f"CHECK nesting: all {len(ids_025):,} patients of the 0.25 subset are in the 0.5 subset ({len(ids_050):,})  OK")

    for frac, expected in _EXPECTED_SUBSET_SIZE.items():
        n = len(subsets[frac])
        assert n == expected, f"subset at fraction {frac} has {n} patients, train.py printed {expected}"
        print(f"CHECK subset size: fraction {frac} gives {n:,} patients, train.py printed {expected:,}  OK")

    return train_part, subsets


# ---------------------------------------------------------------------------
# Small loaders
# ---------------------------------------------------------------------------

def load_transformer_fn(checkpoint_path: Path, device: torch.device, vocab: list[str]) -> PredictFn:
    """Load a checkpoint the way run_eval.py does and wrap it with make_predict_fn."""
    model = HealthTransformer()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    return make_predict_fn(model, device, vocab)


def best_epoch(log_path: Path) -> int:
    """Epoch with the lowest dev NLL in a train log; the checkpoint saved is from that epoch."""
    log = pd.read_csv(log_path)
    return int(log.loc[log["dev_nll"].idxmin(), "epoch"])


def require_files() -> None:
    """Stop before evaluating anything if a needed input is missing."""
    needed = [
        OUTPUTS_DIR / "model_comparison.csv",
        OUTPUTS_DIR / "train_log.csv",
        OUTPUTS_DIR / "analysis_summary.json",
        OUTPUTS_DIR / "transformer_per_disease_auc.csv",
    ]
    for tag in list(_SEED_TAGS.values()) + list(_FRAC_TAGS.values()):
        needed.append(CHECKPOINT_DIR / f"{tag}.pt")
        needed.append(OUTPUTS_DIR / f"train_log_{tag}.csv")
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(missing))


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def run_seeds(
    val_sequences: list[Sequence],
    vocab: list[str],
    unseen: set[int],
    device: torch.device,
    committed: pd.Series,
    n_diseases_committed: int,
) -> pd.DataFrame:
    """Evaluate the extra seeds; return one row per seed, then mean, min and max rows.

    Args:
        committed:            the transformer row of model_comparison.csv (seed 42).
        n_diseases_committed: number of diseases in the committed mean AUROC; every
                              new evaluation must average over the same number.
    """
    rows: list[dict] = [{
        "seed": str(_COMMITTED_SEED),
        "mean_nll": float(committed["mean_nll"]),
        "top20_acc": float(committed["top20_acc"]),
        "mean_auroc": float(committed["mean_auroc"]),
        "best_epoch": best_epoch(OUTPUTS_DIR / "train_log.csv"),
    }]
    for seed, tag in _SEED_TAGS.items():
        fn = load_transformer_fn(CHECKPOINT_DIR / f"{tag}.pt", device, vocab)
        r = evaluate(fn, val_sequences, vocab, f"transformer_{tag}", excluded_from_auroc=unseen)
        assert r["n_diseases_auroc"] == n_diseases_committed, "mean AUROC is over a different disease set"
        rows.append({
            "seed": str(seed),
            "mean_nll": r["mean_nll"],
            "top20_acc": r["top20_acc"],
            "mean_auroc": r["mean_auroc"],
            "best_epoch": best_epoch(OUTPUTS_DIR / f"train_log_{tag}.csv"),
        })

    per_seed = pd.DataFrame(rows)
    numeric = per_seed.drop(columns=["seed"])
    stats = pd.DataFrame(
        [numeric.mean(), numeric.min(), numeric.max()],
        index=["mean", "min", "max"],
    ).rename_axis("seed").reset_index()
    return pd.concat([per_seed, stats], ignore_index=True)


# ---------------------------------------------------------------------------
# Data efficiency
# ---------------------------------------------------------------------------

def run_data_efficiency(
    val_sequences: list[Sequence],
    vocab: list[str],
    unseen: set[int],
    device: torch.device,
    sex_indices: set[int],
    subsets: dict[float, list[Sequence]],
    age_sex_K: float,
    committed: pd.Series,
    n_diseases_committed: int,
) -> pd.DataFrame:
    """Score both models at each training fraction of train_part.

    Transformer: 0.25 and 0.5 from the tagged checkpoints; 1.0 is the committed run,
    read from model_comparison.csv.  age_sex: refit on the same subset at every
    fraction, including 1.0 (all of train_part), with K fixed at the value tuned at
    full data.

    Returns:
        DataFrame with columns fraction, model, mean_nll, mean_auroc, n_train_patients.
    """
    rows: list[dict] = []
    for frac in (0.25, 0.5, 1.0):
        subset = subsets[frac]
        frac_tag = _FRAC_TAGS.get(frac, "frac100")

        if frac in _FRAC_TAGS:
            fn = load_transformer_fn(CHECKPOINT_DIR / f"{frac_tag}.pt", device, vocab)
            r = evaluate(fn, val_sequences, vocab, f"transformer_{frac_tag}", excluded_from_auroc=unseen)
            assert r["n_diseases_auroc"] == n_diseases_committed, "mean AUROC is over a different disease set"
            t_nll, t_auc = r["mean_nll"], r["mean_auroc"]
        else:
            t_nll, t_auc = float(committed["mean_nll"]), float(committed["mean_auroc"])
        rows.append({
            "fraction": frac, "model": "transformer",
            "mean_nll": t_nll, "mean_auroc": t_auc, "n_train_patients": len(subset),
        })

        age_sex_fn = fit_age_sex(subset, vocab, sex_indices, K=age_sex_K)
        r = evaluate(age_sex_fn, val_sequences, vocab, f"age_sex_{frac_tag}", excluded_from_auroc=unseen)
        assert r["n_diseases_auroc"] == n_diseases_committed, "mean AUROC is over a different disease set"
        rows.append({
            "fraction": frac, "model": "age_sex",
            "mean_nll": r["mean_nll"], "mean_auroc": r["mean_auroc"], "n_train_patients": len(subset),
        })
    return pd.DataFrame(rows)


def plot_data_efficiency(df: pd.DataFrame, age_sex_K: float) -> None:
    """Mean AUROC against training fraction, one line per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, style in (("transformer", "-o"), ("age_sex", "--s")):
        sub = df[df["model"] == m].sort_values("fraction")
        ax.plot(sub["fraction"], sub["mean_auroc"], style, label=m)
    ax.set_xticks(sorted(df["fraction"].unique()))
    ax.set_xlabel("Fraction of train_part used for training")
    ax.set_ylabel("Mean per-disease AUROC on val")
    ax.set_title(
        "Data efficiency\n"
        f"one training run per fraction; age_sex K = {age_sex_K:g}, tuned at full data, not retuned",
        fontsize=10,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "data_efficiency.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    require_files()

    train_df = load_split("train")
    val_df = load_split("val")
    vocab = load_vocab()
    train_df = train_df[train_df["person_id"] != _EXCLUDED_TRAIN_PATIENT]
    train_sequences = build_sequences(train_df)
    val_sequences = build_sequences(val_df)
    sex_indices = {i for i, v in enumerate(vocab) if v in ("Female", "Male")}
    # Same exclusion set for every run, computed from full train as in run_eval.py, so every
    # run is scored on the same eligible diseases and the mean AUROCs are comparable.
    unseen: set[int] = set(val_df["token"]) - set(train_df["token"])

    # build_sequences returns patients in ascending person_id order
    person_ids = np.sort(train_df["person_id"].unique()).tolist()
    assert len(person_ids) == len(train_sequences)
    id_of = {id(tokens): pid for (tokens, _), pid in zip(train_sequences, person_ids)}

    train_part, subsets = check_splits(train_sequences, id_of)

    summary_in = json.loads((OUTPUTS_DIR / "analysis_summary.json").read_text())
    age_sex_K = float(summary_in["provenance"]["age_sex_K"])
    print(
        f"age_sex K = {age_sex_K:g} for every fraction. K was tuned at full data"
        " (dev split of the full train set) and is not retuned per fraction."
    )

    mc = pd.read_csv(OUTPUTS_DIR / "model_comparison.csv").set_index("name")
    committed = mc.loc["transformer"]
    n_diseases_committed = len(pd.read_csv(OUTPUTS_DIR / "transformer_per_disease_auc.csv"))

    device = select_device()
    print(f"Device: {device}")

    seeds_df = run_seeds(val_sequences, vocab, unseen, device, committed, n_diseases_committed)
    eff_df = run_data_efficiency(
        val_sequences, vocab, unseen, device, sex_indices, subsets, age_sex_K,
        committed, n_diseases_committed,
    )

    seeds_df.to_csv(OUTPUTS_DIR / "seeds_summary.csv", index=False)
    eff_df.to_csv(OUTPUTS_DIR / "data_efficiency.csv", index=False)
    plot_data_efficiency(eff_df, age_sex_K)

    print("\n=== Seeds (transformer, full train_part) ===")
    print(seeds_df.to_string(index=False, float_format="{:.4f}".format))
    print("\n=== Data efficiency (fraction of train_part; age_sex K tuned at full data) ===")
    print(eff_df.to_string(index=False, float_format="{:.4f}".format))
    for name in ("seeds_summary.csv", "data_efficiency.csv", "data_efficiency.png"):
        print(f"Saved {OUTPUTS_DIR / name}")
