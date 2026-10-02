"""Train HealthTransformer with early stopping on dev NLL."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data import build_sequences, load_split, load_vocab, split_train_dev
from src.model import HealthTransformer, select_device

OUTPUTS_DIR = Path("outputs")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"

SEED: int = 42
BATCH_SIZE: int = 64
LR: float = 3e-4
WEIGHT_DECAY: float = 0.01
MAX_EPOCHS: int = 30
PATIENCE: int = 4
_EXCLUDED_TRAIN_PATIENT: int = 402867


class SequenceDataset(Dataset):
    """Pads each sequence to block_size; converts ages from days to years."""

    def __init__(
        self,
        sequences: list[tuple[np.ndarray, np.ndarray]],
        block_size: int,
    ) -> None:
        self.sequences = sequences
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, ages = self.sequences[idx]
        L = min(len(tokens), self.block_size)
        tok = np.zeros(self.block_size, dtype=np.int64)
        age = np.zeros(self.block_size, dtype=np.float32)
        tok[:L] = tokens[:L]
        age[:L] = ages[:L] / 365.25  # days → years, matching AgeEncoder's expectation
        return torch.from_numpy(tok), torch.from_numpy(age)


def _train_epoch(
    model: HealthTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One training pass; returns mean NLL per non-padding token."""
    model.train()
    total_nll = 0.0
    n_tokens = 0

    for tokens, ages in loader:
        tokens = tokens.to(device)  # (B, T) int64
        ages = ages.to(device)      # (B, T) float32, years

        pad_mask = tokens == 0      # (B, T) True = padding key to ignore
        logits = model(tokens, ages, key_padding_mask=pad_mask)  # (B, T, V)

        # Next-token shift: logits[:, 0..T-2, :] compared against tokens[:, 1..T-1].
        # At position k-1 the model predicts t_k, using only t_0..t_{k-1} and
        # a_0..a_{k-1} (causal mask).  Position 0 (sex token) is never a target.
        # Padding targets are excluded via ignore_index=0.
        logits_shift = logits[:, :-1, :].contiguous().view(-1, HealthTransformer.VOCAB_SIZE)
        targets_shift = tokens[:, 1:].contiguous().view(-1)

        n_valid = int((targets_shift != 0).sum())
        if n_valid == 0:
            continue

        loss = F.cross_entropy(logits_shift, targets_shift, ignore_index=0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_nll += loss.item() * n_valid
        n_tokens += n_valid

    return total_nll / n_tokens if n_tokens > 0 else float("nan")


def _eval_nll(
    model: HealthTransformer,
    sequences: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> float:
    """Mean NLL over all prediction points using teacher forcing.

    Teacher-forcing NLL equals the prefix-by-prefix NLL in evaluate.py: the
    causal mask ensures the output at position k-1 depends only on inputs at
    positions 0..k-1, identical to running the model on a truncated prefix.
    This avoids the overhead of expanding each patient into L-1 individual
    prefixes during training.
    """
    model.eval()
    loader = DataLoader(
        SequenceDataset(sequences, HealthTransformer.BLOCK_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    total_nll = 0.0
    n_tokens = 0

    with torch.no_grad():
        for tokens, ages in loader:
            tokens = tokens.to(device)
            ages = ages.to(device)
            pad_mask = tokens == 0
            logits = model(tokens, ages, key_padding_mask=pad_mask)

            # Next-token shift: logits[:, 0..T-2, :] compared against tokens[:, 1..T-1].
            # At position k-1 the model predicts t_k, using only t_0..t_{k-1} and
            # a_0..a_{k-1} (causal mask).  Position 0 (sex token) is never a target.
            # Padding targets are excluded via ignore_index=0.
            logits_shift = logits[:, :-1, :].contiguous().view(-1, HealthTransformer.VOCAB_SIZE)
            targets_shift = tokens[:, 1:].contiguous().view(-1)

            n_valid = int((targets_shift != 0).sum())
            if n_valid == 0:
                continue

            loss = F.cross_entropy(logits_shift, targets_shift, ignore_index=0)
            total_nll += loss.item() * n_valid
            n_tokens += n_valid

    return total_nll / n_tokens if n_tokens > 0 else float("nan")


if __name__ == "__main__":
    print(f"Random seed: {SEED}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = select_device()
    print(f"Device: {device}")

    # Data
    train_df = load_split("train")
    vocab = load_vocab()
    train_df = train_df[train_df["person_id"] != _EXCLUDED_TRAIN_PATIENT]
    train_sequences = build_sequences(train_df)

    # Same frac=0.1, seed=0 split as src/baselines.py — dev patients are identical
    train_part, dev_part = split_train_dev(train_sequences)
    print(f"Train patients: {len(train_part):,}  Dev patients: {len(dev_part):,}")

    train_loader = DataLoader(
        SequenceDataset(train_part, HealthTransformer.BLOCK_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = HealthTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    best_dev_nll = float("inf")
    patience_counter = 0
    log_rows: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        try:
            train_loss = _train_epoch(model, train_loader, optimizer, device)
            dev_nll = _eval_nll(model, dev_part, device)
        except RuntimeError as exc:
            if device.type != "mps":
                raise
            # MPS op failure: move everything to CPU and retry this epoch.
            # Optimizer momentum buffers are reset (minor setback on one epoch).
            print(f"MPS error at epoch {epoch} ({exc}); falling back to CPU")
            device = torch.device("cpu")
            model = model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=MAX_EPOCHS - epoch + 1
            )
            train_loss = _train_epoch(model, train_loader, optimizer, device)
            dev_nll = _eval_nll(model, dev_part, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"epoch {epoch:3d}/{MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"dev_nll={dev_nll:.4f}  "
            f"elapsed={elapsed:.1f}s"
        )
        log_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "dev_nll": dev_nll, "elapsed_sec": round(elapsed, 1)}
        )

        if dev_nll < best_dev_nll:
            best_dev_nll = dev_nll
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\nBest dev NLL: {best_dev_nll:.4f}  Checkpoint: {CHECKPOINT_PATH}")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(OUTPUTS_DIR / "train_log.csv", index=False)
    print(f"Saved {OUTPUTS_DIR}/train_log.csv")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ep = log_df["epoch"]
    ax1.plot(ep, log_df["train_loss"], color="tab:blue", label="train loss")
    ax2.plot(ep, log_df["dev_nll"], color="tab:orange", linestyle="--", label="dev NLL")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss (NLL)", color="tab:blue")
    ax2.set_ylabel("Dev NLL", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax1.set_title("Training curve")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "train_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUTS_DIR}/train_curve.png")
