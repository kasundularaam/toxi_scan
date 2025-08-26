# train_word_model_imbalanced.py
import pandas as pd
import numpy as np
import unicodedata
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, auc
from joblib import dump
from pathlib import Path

DATA_CSV = "data/words.csv"
MODEL_OUT = "models/word_cuss_lr.joblib"
THRESH_OUT = "models/threshold.txt"

ZWS = re.compile(r"[\u200B-\u200D\uFEFF]")
REPEATS = re.compile(r"(.)\1{2,}")


def normalize(word: str) -> str:
    w = unicodedata.normalize("NFKC", str(word)).lower()
    w = ZWS.sub("", w)
    w = REPEATS.sub(r"\1\1", w)
    return w.strip()


# --- load & prep
df = pd.read_csv(DATA_CSV)
df["word"] = df["word"].astype(str).map(normalize)
df = df.drop_duplicates(subset=["word"]).reset_index(drop=True)

X = df["word"].tolist()
y = df["label"].astype(int).values

# --- split (stratified)
Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# --- model (char n-grams + LR with class_weight)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char",
     ngram_range=(2, 5), min_df=1, max_features=100_000)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

pipe.fit(Xtr, ytr)

# --- validation
probs = pipe.predict_proba(Xva)[:, 1]
preds = (probs >= 0.5).astype(int)
print("== Report @ 0.5 ==")
print(classification_report(yva, preds,
      target_names=["clean", "cuss"], digits=4))

# --- find best threshold (macro-F1 or F1 for 'cuss')
prec, rec, th = precision_recall_curve(yva, probs)
f1s = 2*prec*rec / (prec+rec + 1e-12)
best_idx = np.argmax(f1s)
best_thr = float(th[best_idx-1]) if best_idx > 0 else 0.5

print(f"Best threshold (F1 on 'cuss'): {best_thr:.3f}")
print(f"Validation AUPRC: {auc(rec, prec):.4f}")

# --- save
Path("models").mkdir(exist_ok=True)
dump(pipe, MODEL_OUT)
with open(THRESH_OUT, "w") as f:
    f.write(str(best_thr))
print(f"Saved -> {MODEL_OUT} and {THRESH_OUT}")
