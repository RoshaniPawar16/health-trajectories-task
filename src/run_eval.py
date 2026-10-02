"""Load best checkpoint, evaluate on val, compare with baseline numbers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.data import build_sequences, load_split, load_vocab
from src.evaluate import evaluate
from src.model import HealthTransformer, make_predict_fn, select_device

CHECKPOINT_PATH = Path("checkpoints/best.pt")
_EXCLUDED_TRAIN_PATIENT: int = 402867

_COMPARE_COLS = [
    "name", "mean_nll", "top1_acc", "top5_acc", "top20_acc",
    "mean_auroc", "median_auroc",
]

if __name__ == "__main__":
    train_df = load_split("train")
    val_df = load_split("val")
    vocab = load_vocab()

    # Drop 402867 only to compute the unseen token set consistently with baselines
    train_df = train_df[train_df["person_id"] != _EXCLUDED_TRAIN_PATIENT]
    val_sequences = build_sequences(val_df)
    unseen: set[int] = set(val_df["token"]) - set(train_df["token"])

    device = select_device()
    print(f"Device: {device}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found — run python -m src.train first"
        )

    model = HealthTransformer()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)

    predict_fn = make_predict_fn(model, device, vocab)
    transformer_result = evaluate(
        predict_fn,
        val_sequences,
        vocab,
        "transformer",
        excluded_from_auroc=unseen,
    )

    # Print comparison table
    baseline_path = Path("outputs/baselines_summary.csv")
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)[_COMPARE_COLS]
    else:
        print("(outputs/baselines_summary.csv not found; showing transformer only)")
        baseline_df = pd.DataFrame(columns=_COMPARE_COLS)

    transformer_row = pd.DataFrame([{c: transformer_result[c] for c in _COMPARE_COLS}])
    comparison = pd.concat([baseline_df, transformer_row], ignore_index=True)

    print("\n=== Comparison: baselines vs transformer ===")
    print(comparison.to_string(index=False, float_format="{:.4f}".format))
