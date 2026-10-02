"""Same-age tie check. Run from repo root: python check_ties.py"""
import numpy as np
from src.data import build_sequences, load_split, load_vocab

vocab = load_vocab()
sex = {i for i, v in enumerate(vocab) if v in ("Female", "Male")}

for split in ("train", "val"):
    df = load_split(split)
    if split == "train":
        df = df[df["person_id"] != 402867]
    n_pairs = n_tied = n_asc = n_desc = 0
    for toks, ages in build_sequences(df):
        # prediction pairs: history end k-1, target k, for k = 1..L-1
        prev_t, next_t = toks[:-1], toks[1:]
        prev_a, next_a = ages[:-1], ages[1:]
        tied = prev_a == next_a
        diag = ~np.isin(prev_t, list(sex))  # ignore sex token -> first diagnosis
        n_pairs += len(next_t)
        n_tied += int(tied.sum())
        m = tied & diag
        n_asc += int((next_t[m] > prev_t[m]).sum())
        n_desc += int((next_t[m] < prev_t[m]).sum())
    print(f"[{split}] prediction points: {n_pairs:,}")
    print(f"[{split}] target has same age as last history event: {n_tied:,} ({100*n_tied/n_pairs:.2f}%)")
    tot = n_asc + n_desc
    if tot:
        print(f"[{split}] among diagnosis-diagnosis ties: ascending token index {n_asc:,} "
              f"({100*n_asc/tot:.2f}%), descending {n_desc:,} ({100*n_desc/tot:.2f}%)")
    print()

# Is vocab index order the same as alphabetical ICD order?
codes = [v for i, v in enumerate(vocab) if i > 0 and i not in sex]
print("vocab diagnosis codes in alphabetical order:", codes == sorted(codes))
