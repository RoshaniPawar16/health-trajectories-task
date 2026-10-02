"""Small causal transformer for health trajectory modelling.

Architecture follows Delphi-2M (Shmatko et al., Nature 2025) in spirit:
no learned positional embedding; age at each event carries temporal position.
Built from scratch in PyTorch; nothing copied from the Delphi repository.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """Return MPS if available and functional, else CPU."""
    if torch.backends.mps.is_available():
        try:
            d = torch.device("mps")
            torch.zeros(4, 4).to(d).sum()
            return d
        except Exception as e:
            print(f"MPS test failed ({e}); falling back to CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class AgeEncoder(nn.Module):
    """Encodes scalar age (in years) as sinusoidal features, projected to d_model.

    Frequencies mirror the original transformer positional encoding applied to a
    continuous age value rather than a discrete position index.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        half = d_model // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))
        self.register_buffer("freqs", freqs)  # (half,) — not a learned parameter
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, age_years: torch.Tensor) -> torch.Tensor:
        """
        Args:
            age_years: (B, T) float32
        Returns:
            (B, T, d_model) float32
        """
        v = age_years.unsqueeze(-1) * self.freqs        # (B, T, half)
        feats = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B, T, d_model)
        return self.proj(feats)


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular additive float mask; −inf above diagonal forces causal attention."""
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: (LN → MHA → add), (LN → FFN → add)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:               (B, T, d_model)
            attn_mask:       (T, T) additive causal mask — explicit, not is_causal,
                             to avoid MPS-specific behaviour differences
            key_padding_mask: (B, T) float; -inf = padding position to ignore as a key
        Returns:
            (B, T, d_model)
        """
        h, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HealthTransformer(nn.Module):
    """Causal transformer over tokenised health trajectories."""

    VOCAB_SIZE: int = 997
    D_MODEL: int = 128
    N_HEADS: int = 4
    N_LAYERS: int = 4
    DROPOUT: float = 0.1
    BLOCK_SIZE: int = 80  # covers the longest train/val sequence (69 events)

    def __init__(self) -> None:
        super().__init__()
        d = self.D_MODEL
        self.token_emb = nn.Embedding(self.VOCAB_SIZE, d, padding_idx=0)
        self.age_enc = AgeEncoder(d)
        self.drop = nn.Dropout(self.DROPOUT)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, self.N_HEADS, self.DROPOUT) for _ in range(self.N_LAYERS)]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.VOCAB_SIZE, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        ages: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens:           (B, T) int64
            ages:             (B, T) float32, age in years at each position
            key_padding_mask: (B, T) bool or float; True / -inf = padding position
        Returns:
            logits (B, T, VOCAB_SIZE)
        """
        T = tokens.size(1)
        x = self.drop(self.token_emb(tokens) + self.age_enc(ages))
        causal = _causal_mask(T, tokens.device)  # (T, T) float32, -inf above diagonal

        # MHA requires both masks to be the same dtype.  Convert bool key_padding_mask
        # to a float additive mask so it matches causal: True → -inf, False → 0.0.
        if key_padding_mask is not None and key_padding_mask.dtype == torch.bool:
            kpm = torch.zeros_like(key_padding_mask, dtype=causal.dtype)
            key_padding_mask = kpm.masked_fill(key_padding_mask, float("-inf"))

        for block in self.blocks:
            x = block(x, attn_mask=causal, key_padding_mask=key_padding_mask)
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# Predict-fn factory
# ---------------------------------------------------------------------------

def make_predict_fn(
    model: HealthTransformer,
    device: torch.device,
    vocab: list[str],
    batch_size: int = 512,
) -> Callable[[list[tuple[np.ndarray, np.ndarray]]], np.ndarray]:
    """Return a predict_fn compatible with src.evaluate.evaluate.

    Logits for padding (index 0) and both sex tokens are masked to −inf before
    softmax so probability mass matches the baselines exactly (no probability
    ever assigned to tokens that cannot be prediction targets).

    Args:
        model:      HealthTransformer; set to eval mode internally.
        device:     inference device.
        vocab:      list from load_vocab(); used to locate sex token indices.
        batch_size: prefixes per forward pass (controls peak memory).

    Returns:
        Function taking list of (tokens, ages) pairs — ages in days, as produced
        by build_sequences — and returning float32 ndarray (n_prefixes, VOCAB_SIZE)
        with rows summing to 1.
    """
    model.eval()
    BLOCK = HealthTransformer.BLOCK_SIZE
    VSZ = HealthTransformer.VOCAB_SIZE

    _mask_idx = torch.tensor(
        [0] + [i for i, v in enumerate(vocab) if v in ("Female", "Male")],
        dtype=torch.long,
        device=device,
    )

    def predict(prefixes: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        n = len(prefixes)
        if n == 0:
            return np.empty((0, VSZ), dtype=np.float32)

        out = np.empty((n, VSZ), dtype=np.float32)

        for start in range(0, n, batch_size):
            batch = prefixes[start : start + batch_size]
            B = len(batch)

            # Clamp to BLOCK_SIZE keeping from position 0 (preserves the sex token t_0)
            lengths = [min(len(t), BLOCK) for t, _ in batch]
            T = max(lengths)

            tok_np = np.zeros((B, T), dtype=np.int64)
            age_np = np.zeros((B, T), dtype=np.float32)
            for i, (toks, ages) in enumerate(batch):
                L = lengths[i]
                tok_np[i, :L] = toks[:L]
                age_np[i, :L] = ages[:L] / 365.25  # days → years

            tok_t = torch.from_numpy(tok_np).to(device)
            age_t = torch.from_numpy(age_np).to(device)
            pad_mask = tok_t == 0  # (B, T) True = padding key to ignore

            with torch.no_grad():
                logits = model(tok_t, age_t, key_padding_mask=pad_mask)  # (B, T, V)

            # Gather each prefix's prediction: output at the last real position
            last_pos = torch.tensor([l - 1 for l in lengths], device=device)
            last_logits = logits[torch.arange(B, device=device), last_pos, :]  # (B, V)

            # Mask padding and sex tokens — probability mass must never fall on them
            last_logits[:, _mask_idx] = float("-inf")

            probs = torch.softmax(last_logits, dim=-1).cpu().numpy()
            out[start : start + B] = probs

        return out

    return predict
