# predict_word.py
import unicodedata
import regex as re
from joblib import load
from pathlib import Path

MODEL_PATH = "models/word_cuss_lr.joblib"
THRESH_PATH = "models/threshold.txt"

ZWS = re.compile(r"[\u200B-\u200D\uFEFF]")
REPEATS = re.compile(r"(.)\1{2,}")


def normalize(word: str) -> str:
    w = unicodedata.normalize("NFKC", str(word)).lower()
    w = ZWS.sub("", w)
    w = REPEATS.sub(r"\1\1", w)
    return w.strip()


_model = load(MODEL_PATH)
_thr = 0.5
if Path(THRESH_PATH).exists():
    _thr = float(Path(THRESH_PATH).read_text().strip() or 0.5)


def is_cuss(word: str, threshold: float | None = None):
    th = _thr if threshold is None else threshold
    x = [normalize(word)]
    p = float(_model.predict_proba(x)[0][1])
    return {"word": word, "normalized": x[0], "is_cuss": p >= th, "score": p, "threshold": th}


print(is_cuss("Hutta"))          # expect True with a high score
print(is_cuss("Hu—ttaaa"))       # normalization should still catch
print(is_cuss("සුභ"))            # expect False
