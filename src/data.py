"""Data loading, vocabulary, sanity checks, and sequence building for health trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")


def load_split(name: str) -> pd.DataFrame:
    """Load a binary split file into a DataFrame.

    Args:
        name: 'train' or 'val'

    Returns:
        DataFrame with columns person_id, age_days, token (true vocab index), age_years.
    """
    path = DATA_DIR / f"{name}.bin"
    raw = np.fromfile(path, dtype="<u4")
    arr = raw.reshape(-1, 3)
    df = pd.DataFrame(arr, columns=["person_id", "age_days", "token"])
    df["token"] = (df["token"] + 1).astype(np.int32)  # stored as vocab_index - 1
    df["age_years"] = df["age_days"] / 365.25
    return df


def load_vocab() -> list[str]:
    """Load the vocabulary from labels.csv.

    Returns:
        List of token names where list index == vocabulary index.
        Index 0 is the padding token and never appears in sequences.
    """
    path = DATA_DIR / "labels.csv"
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def sanity_checks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocab: list[str],
) -> None:
    """Print summary statistics and PASS/FAIL for five data integrity checks.

    Args:
        train_df: DataFrame returned by load_split('train').
        val_df:   DataFrame returned by load_split('val').
        vocab:    List returned by load_vocab().
    """
    train_counts = train_df.groupby("person_id").size()
    val_counts = val_df.groupby("person_id").size()
    all_ages = pd.concat([train_df["age_years"], val_df["age_years"]])

    print("=== Summary ===")
    print(f"Patients      : train={train_df['person_id'].nunique():,}  val={val_df['person_id'].nunique():,}")
    print(f"Events        : train={len(train_df):,}  val={len(val_df):,}")
    print(
        f"Events/patient: train median={train_counts.median():.1f}  max={train_counts.max()}"
        f"  |  val median={val_counts.median():.1f}  max={val_counts.max()}"
    )
    print(f"Age range     : {all_ages.min():.2f} – {all_ages.max():.2f} years")
    print(f"Vocab size    : {len(vocab)}")
    print()
    print("=== Checks ===")

    # 1. No person_id in both splits
    overlap = set(train_df["person_id"]) & set(val_df["person_id"])
    if not overlap:
        print("PASS  no person_id appears in both splits")
    else:
        print(f"FAIL  {len(overlap):,} person_ids appear in both splits")

    # 2. age_days non-decreasing within every patient
    for split_name, df in [("train", train_df), ("val", val_df)]:
        bad = (
            df.groupby("person_id")["age_days"]
            .apply(lambda s: bool((s.diff().dropna() < 0).any()))
        )
        n_bad = int(bad.sum())
        if n_bad == 0:
            print(f"PASS  [{split_name}] age_days is non-decreasing within every patient")
        else:
            example = bad[bad].index[0]
            print(f"FAIL  [{split_name}] age_days decreases for {n_bad:,} patients (e.g. person_id={example})")

    # 3. Every patient has exactly one sex token and it is their first event
    sex_indices = {i for i, name in enumerate(vocab) if name in ("Female", "Male")}
    for split_name, df in [("train", train_df), ("val", val_df)]:
        result = df.groupby("person_id")["token"].apply(
            lambda s: bool(s.isin(sex_indices).sum() == 1 and s.iloc[0] in sex_indices)
        )
        n_bad = int((~result).sum())
        if n_bad == 0:
            print(f"PASS  [{split_name}] every patient has exactly one sex token as their first event")
        else:
            example = result[~result].index[0]
            print(f"FAIL  [{split_name}] {n_bad:,} patients violate the sex-token rule (e.g. person_id={example})")

    # 4. Every token in val appears in train
    unseen = set(val_df["token"]) - set(train_df["token"])
    if not unseen:
        print("PASS  every token in val appears in train")
    else:
        names = [vocab[i] if i < len(vocab) else str(i) for i in list(unseen)[:3]]
        print(f"FAIL  {len(unseen):,} tokens in val not seen in train (e.g. {names})")

    # 5. No exact duplicate rows
    for split_name, df in [("train", train_df), ("val", val_df)]:
        n_dups = int(df.duplicated().sum())
        if n_dups == 0:
            print(f"PASS  [{split_name}] no exact duplicate rows")
        else:
            print(f"FAIL  [{split_name}] {n_dups:,} exact duplicate rows")


def inspect_patient(df: pd.DataFrame, person_id: int, vocab: list[str]) -> None:
    """Print a patient's full event list as a table.

    Args:
        df:        DataFrame from load_split.
        person_id: Patient to inspect.
        vocab:     List returned by load_vocab().
    """
    group = (
        df[df["person_id"] == person_id]
        .sort_values("age_days")
        .reset_index(drop=True)
    )
    print(f"\nPatient {person_id}  ({len(group)} events)")
    print(f"{'pos':>4}  {'age_years':>9}  token")
    print("-" * 36)
    for pos, row in group.iterrows():
        tok = int(row["token"])
        name = vocab[tok] if tok < len(vocab) else str(tok)
        print(f"{pos:>4}  {row['age_years']:>9.2f}  {name}")


def unseen_token_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocab: list[str],
) -> None:
    """Print val tokens absent from train, affected event counts, and token names.

    Args:
        train_df: DataFrame returned by load_split('train').
        val_df:   DataFrame returned by load_split('val').
        vocab:    List returned by load_vocab().
    """
    unseen = set(val_df["token"]) - set(train_df["token"])
    n_unseen = len(unseen)
    n_affected = int(val_df["token"].isin(unseen).sum())
    pct = 100.0 * n_affected / len(val_df)
    names = sorted(vocab[i] if i < len(vocab) else str(i) for i in unseen)

    print("\n=== Unseen token report ===")
    print(f"Val tokens absent from train       : {n_unseen}")
    print(f"Val events carrying an unseen token: {n_affected:,} ({pct:.2f}% of val events)")
    print(f"Token names ({n_unseen} total, alphabetical):")
    for name in names:
        print(f"  {name}")


def build_sequences(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build per-patient token and age sequences sorted by age_days.

    Args:
        df: DataFrame from load_split.

    Returns:
        List of (tokens, ages) tuples — one per patient — where tokens is int32
        and ages is float32, both in ascending age_days order.
    """
    sequences: list[tuple[np.ndarray, np.ndarray]] = []
    for _, group in df.sort_values(["person_id", "age_days"]).groupby("person_id", sort=False):
        tokens = group["token"].to_numpy(dtype=np.int32)
        ages = group["age_days"].to_numpy(dtype=np.float32)
        sequences.append((tokens, ages))
    return sequences


def split_train_dev(
    train_sequences: list[tuple[np.ndarray, np.ndarray]],
    frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Hold out frac of patients as a dev set for hyperparameter tuning.

    Split is by patient index (one sequence per patient in the list).
    Fixed frac=0.1 and seed=0 are reused for baseline K tuning and model
    early stopping — always call with the same defaults to get the same split.

    Args:
        train_sequences: output of build_sequences(train_df).
        frac:  fraction of patients to hold out (default 0.1).
        seed:  RNG seed for reproducibility (default 0).

    Returns:
        (train_part, dev_part) as lists of (tokens, ages) tuples.
    """
    rng = np.random.default_rng(seed)
    n = len(train_sequences)
    perm = rng.permutation(n)
    n_dev = round(n * frac)
    dev_set = set(perm[:n_dev].tolist())
    train_part = [seq for i, seq in enumerate(train_sequences) if i not in dev_set]
    dev_part = [seq for i, seq in enumerate(train_sequences) if i in dev_set]
    return train_part, dev_part


if __name__ == "__main__":
    train_df = load_split("train")
    val_df = load_split("val")
    vocab = load_vocab()

    sanity_checks(train_df, val_df, vocab)

    # Inspect the first train patient that fails the sex-token check
    sex_indices = {i for i, name in enumerate(vocab) if name in ("Female", "Male")}
    sex_check = train_df.groupby("person_id")["token"].apply(
        lambda s: bool(s.isin(sex_indices).sum() == 1 and s.iloc[0] in sex_indices)
    )
    failing = sex_check[~sex_check].index.tolist()
    if failing:
        inspect_patient(train_df, failing[0], vocab)

    unseen_token_report(train_df, val_df, vocab)

    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Figure 1: events per patient
    train_counts = train_df.groupby("person_id").size()
    val_counts = val_df.groupby("person_id").size()

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(
        0,
        max(train_counts.max(), val_counts.max()),
        60,
    )
    ax.hist(train_counts, bins=bins, alpha=0.6, label="train")
    ax.hist(val_counts, bins=bins, alpha=0.6, label="val")
    ax.set_xlabel("Events per patient")
    ax.set_ylabel("Patients")
    ax.set_title("Events per patient distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "events_per_patient.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {OUTPUTS_DIR}/events_per_patient.png")

    # Figure 2: age at event
    fig, ax = plt.subplots(figsize=(8, 4))
    max_age = max(train_df["age_years"].max(), val_df["age_years"].max())
    bins = np.linspace(0, max_age, 80)
    ax.hist(train_df["age_years"], bins=bins, alpha=0.6, label="train")
    ax.hist(val_df["age_years"], bins=bins, alpha=0.6, label="val")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Events")
    ax.set_title("Age at event distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "age_at_event.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUTS_DIR}/age_at_event.png")
