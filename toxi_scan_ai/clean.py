import csv
import json
import os
import unicodedata
from pathlib import Path
import regex as re
import pandas as pd

# ------------ config ------------
NORMAL_CSV = "normal.csv"
CUSS_JSON  = "cuss.json"
OUT_CSV    = "data/words.csv"
MIN_LEN    = 2   # ignore 1-char tokens like punctuation
# --------------------------------

Path("data").mkdir(exist_ok=True)

# Basic normalization (Sinhala + Singlish friendly)
ZWS = re.compile(r"[\u200B-\u200D\uFEFF]")
REPEATS = re.compile(r"(.)\1{2,}")             # collapse loooong runs
NON_WORD = re.compile(r"[^\p{L}\p{N}]+", re.UNICODE)  # remove punctuation chunks
UNKNOWN = re.compile(r"^\[unk?own\]$", re.IGNORECASE)  # [Unkown]/[Unknown] variants

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = ZWS.sub("", s)
    s = REPEATS.sub(r"\1\1", s)  # keep at most double repeats
    return s

def tokenize_line(s: str):
    # 1) normalize, 2) replace punctuation with space, 3) split
    s = norm(s)
    s = NON_WORD.sub(" ", s)
    toks = [t for t in s.split() if len(t) >= MIN_LEN and not UNKNOWN.match(t)]
    return toks

# -------- read normal.csv (Sinhala + Singlish only) --------
# Expected columns: Sinhala, English, Singlish
norm_df = pd.read_csv(NORMAL_CSV)
candidate_cols = [c for c in norm_df.columns if c.lower() in ("sinhala", "singlish")]
if not candidate_cols:
    raise RuntimeError("normal.csv must have 'Sinhala' and/or 'Singlish' columns.")

normal_words = set()
for col in candidate_cols:
    for val in norm_df[col].dropna().astype(str):
        for w in tokenize_line(val):
            normal_words.add(w)

# -------- read cuss.json (patterns -> words) --------
with open(CUSS_JSON, "r", encoding="utf-8") as f:
    cj = json.load(f)

cuss_words = set()
for item in cj.get("patterns", []):
    pat = item.get("pattern", "")
    w = norm(pat)
    # also strip punctuation for patterns like "කැriයා" → split into sensible pieces but keep base token
    # keep the raw normalized token as-is
    if len(w) >= MIN_LEN and not UNKNOWN.match(w):
        cuss_words.add(w)

# -------- remove obvious overlaps (if any) --------
# If a word appears in both sets, keep it as cuss (positive) to be safe
overlap = normal_words & cuss_words
if overlap:
    normal_words -= overlap  # prefer positive label for overlaps

# -------- write out data/words.csv --------
rows = []
rows += [{"word": w, "label": 0} for w in sorted(normal_words)]
rows += [{"word": w, "label": 1} for w in sorted(cuss_words)]

out_df = pd.DataFrame(rows)
out_df.drop_duplicates(subset=["word"], keep="first", inplace=True)
out_df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

print(f"Wrote {len(out_df)} rows to {OUT_CSV}")
print(f"  normal: {len(normal_words)} | cuss: {len(cuss_words)} | overlap_forced_positive: {len(overlap)}")
