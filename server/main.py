# main.py
import os
import json
import unicodedata
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional

import regex as re
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load as joblib_load

# =========================================================
# Env / Config
# =========================================================
load_dotenv()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "false").lower() in ("1", "true", "yes", "on")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

DETECTOR = os.getenv("DETECTOR", "ai").lower()  # "ai" | "rules" | "hybrid"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# --- Gemini (OCR) OPTIONAL ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLIENT = None
try:
    if GEMINI_API_KEY:
        from google import genai
        from google.genai import types as gx
        CLIENT = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    CLIENT = None  # if SDK not installed or key invalid

# =========================================================
# App
# =========================================================
app = FastAPI(title="ToxiScan (AI Word Service)", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Shared tagging utils
# =========================================================


def merge_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged: List[List[Any]] = []
    for s, e, lbl in spans:
        if not merged:
            merged.append([s, e, lbl])
        else:
            ls, le, _ = merged[-1]
            if s <= le:
                if e > le:
                    merged[-1][1] = e
            else:
                merged.append([s, e, lbl])
    return [(s, e, lbl) for s, e, lbl in merged]


def tag_text(text: str, spans: List[Tuple[int, int, str]]) -> str:
    if not spans:
        return text
    out, last = [], 0
    for s, e, _ in spans:
        out.append(text[last:s])
        out.append("<cuss>")
        out.append(text[s:e])
        out.append("</cuss>")
        last = e
    out.append(text[last:])
    return "".join(out)


# =========================================================
# AI word-level model loader
# =========================================================
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "word_cuss_lr.joblib"
THRESH_PATH = MODEL_DIR / "threshold.txt"

_AI_MODEL = None
_AI_THR = 0.5


def _normalize_word(w: str) -> str:
    w = unicodedata.normalize("NFKC", str(w)).lower().strip()
    # remove zero-width chars and compress long repeats
    w = re.sub(r"[\u200B-\u200D\uFEFF]", "", w)
    w = re.sub(r"(.)\1{2,}", r"\1\1", w)
    return w


# token pattern: letters/digits; allow -, ., and ZWJ inside tokens
TOKEN_RE = re.compile(
    r"\b[\p{L}\p{N}][\p{L}\p{N}\-._\u200B-\u200D]*\b", re.UNICODE)


def _load_ai_model():
    global _AI_MODEL, _AI_THR
    if MODEL_PATH.exists():
        _AI_MODEL = joblib_load(MODEL_PATH)
    if THRESH_PATH.exists():
        try:
            _AI_THR = float(THRESH_PATH.read_text().strip())
        except Exception:
            _AI_THR = 0.5


_load_ai_model()


@lru_cache(maxsize=20000)
def _score_norm(norm_word: str) -> float:
    if _AI_MODEL is None or not norm_word:
        return 0.0
    try:
        return float(_AI_MODEL.predict_proba([norm_word])[0][1])
    except Exception:
        return 0.0


def is_cuss_ai(word: str, threshold: Optional[float] = None) -> Tuple[bool, float, str]:
    thr = _AI_THR if threshold is None else float(threshold)
    nw = _normalize_word(word)
    p = _score_norm(nw)
    return (p >= thr), p, nw


def find_matches_ai(text: str):
    spans, info = [], []
    for m in TOKEN_RE.finditer(text):
        token = m.group(0)
        ok, score, normed = is_cuss_ai(token)
        if ok:
            spans.append((m.start(), m.end(), "ai-cuss"))
            info.append({
                "label": "ai-cuss",
                "match": token,
                "normalized": normed,
                "score": score,
                "start": m.start(),
                "end": m.end()
            })
    merged = merge_spans(spans)
    return tag_text(text, merged), info


# =========================================================
# (Optional) Rules loader (patterns.json)
# =========================================================
PATTERNS_FILE_ENV = os.getenv("PATTERNS_FILE")
LEET_MAP = {"a": "[a@]", "i": "[i1!|]", "o": "[o0]", "s": "[s$5]",
            "t": "[t7]", "e": "[e3]", "g": "[g9]", "b": "[b8]"}
NOISE = r"(?:[^\p{L}\p{N}]{0,3})"


def _require_patterns(items: Any) -> List[dict]:
    if not isinstance(items, list) or not items:
        raise RuntimeError(
            "No patterns found. Ensure your JSON has a non-empty 'patterns' list.")
    for i, p in enumerate(items):
        if not isinstance(p, dict) or "label" not in p or "pattern" not in p:
            raise RuntimeError(f"Invalid pattern at index {i}.")
        if not isinstance(p["label"], str) or not isinstance(p["pattern"], str) or not p["pattern"].strip():
            raise RuntimeError(f"Bad 'label'/'pattern' at index {i}.")
    return items


def is_latin_letter(ch: str) -> bool:
    o = ord(ch)
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z") or (0x00C0 <= o <= 0x024F)


def _latin_char_class(ch: str) -> str:
    low = ch.lower()
    return LEET_MAP.get(low, re.escape(ch))


def build_pattern(token: str) -> re.Pattern:
    parts = []
    for ch in token:
        if is_latin_letter(ch):
            parts.append(f"(?i:(?:{_latin_char_class(ch)})+)")
        elif ch.isdigit():
            parts.append(r"(?:\d+)")
        else:
            parts.append(f"(?:{re.escape(ch)})+")
    body = NOISE.join(parts)
    return re.compile(body, re.UNICODE)


def load_patterns_if_any() -> Tuple[List[dict], List[dict]]:
    candidates: List[Path] = []
    if PATTERNS_FILE_ENV:
        candidates.append(Path(PATTERNS_FILE_ENV))
    candidates.append(Path.cwd() / "patterns.json")
    candidates.append(Path(__file__).parent / "patterns.json")
    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items = _require_patterns(data.get("patterns"))
            compiled = [{"label": i["label"], "pattern": i["pattern"],
                         "rx": build_pattern(i["pattern"])} for i in items]
            print(f"[patterns] loaded {len(items)} from {p}")
            return items, compiled
    return [], []  # none found


PATTERNS_RAW, COMPILED = load_patterns_if_any()


def find_matches_rules(text: str):
    spans, info = [], []
    for entry in COMPILED:
        rx, label = entry["rx"], entry["label"]
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), label))
            info.append({
                "label": label,
                "match": text[m.start():m.end()],
                "start": m.start(),
                "end": m.end()
            })
    merged = merge_spans(spans)
    return tag_text(text, merged), info


def find_matches_hybrid(text: str):
    tagged_r, info_r = find_matches_rules(text) if COMPILED else (text, [])
    tagged_a, info_a = find_matches_ai(text)
    spans = []
    for x in info_r:
        spans.append((x["start"], x["end"], x["label"]))
    for x in info_a:
        spans.append((x["start"], x["end"], x["label"]))
    merged = merge_spans(spans)
    tagged = tag_text(text, merged)
    return tagged, (info_r + info_a)


def _detect(text: str):
    if DETECTOR == "ai" or not COMPILED:
        return find_matches_ai(text)
    if DETECTOR == "hybrid":
        return find_matches_hybrid(text)
    return find_matches_rules(text)

# =========================================================
# OCR (optional)
# =========================================================


class OcrResult(BaseModel):
    text: str


def ocr_image_to_text(image_bytes: bytes, mime: str) -> str:
    if CLIENT is None:
        raise RuntimeError(
            "OCR not available: Gemini client is not configured.")
    resp = CLIENT.models.generate_content(
        model=GEMINI_MODEL,
        contents=[gx.Part.from_bytes(data=image_bytes, mime_type=mime),
                  "Extract ALL text exactly as visible. Preserve newlines. Return JSON with a single field 'text'."],
        config={"response_mime_type": "application/json",
                "response_schema": OcrResult, "temperature": 0},
    )
    if getattr(resp, "parsed", None) and getattr(resp.parsed, "text", None):
        return resp.parsed.text
    return json.loads(resp.text)["text"]

# =========================================================
# API Models
# =========================================================


class AnalyzeTextRequest(BaseModel):
    text: str
    threshold: Optional[float] = None  # allow per-request override


class AnalyzeResponse(BaseModel):
    source: str
    raw_text: str
    tagged_text: str
    matches: List[Dict[str, Any]]

# =========================================================
# Routes
# =========================================================


@app.get("/health")
def health():
    return {
        "ok": True,
        "detector": DETECTOR,
        "ai_loaded": bool(_AI_MODEL),
        "ai_threshold": _AI_THR,
        "patterns": len(COMPILED),
        "ocr_enabled": CLIENT is not None,
    }


@app.get("/is_cuss")
def is_cuss(word: str = Query(..., min_length=1), threshold: Optional[float] = None):
    ok, score, normed = is_cuss_ai(word, threshold=threshold)
    return {"word": word, "normalized": normed, "is_cuss": ok, "score": score, "threshold": threshold or _AI_THR}


@app.post("/analyze/text", response_model=AnalyzeResponse)
async def analyze_text(req: AnalyzeTextRequest = Body(...)):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Provide non-empty 'text'.")
    # allow per-request threshold override
    if req.threshold is not None:
        # temporarily create a local detector with provided threshold
        spans, info = [], []
        for m in TOKEN_RE.finditer(req.text):
            token = m.group(0)
            ok, score, normed = is_cuss_ai(token, threshold=req.threshold)
            if ok:
                spans.append((m.start(), m.end(), "ai-cuss"))
                info.append({"label": "ai-cuss", "match": token, "normalized": normed, "score": score,
                             "start": m.start(), "end": m.end()})
        merged = merge_spans(spans)
        tagged = tag_text(req.text, merged)
        return JSONResponse(AnalyzeResponse(source="text", raw_text=req.text, tagged_text=tagged, matches=info).model_dump())

    tagged, matches = _detect(req.text)
    return JSONResponse(AnalyzeResponse(source="text", raw_text=req.text, tagged_text=tagged, matches=matches).model_dump())


@app.post("/analyze/image", response_model=AnalyzeResponse)
async def analyze_image(image: UploadFile = File(...)):
    mime = image.content_type or "image/jpeg"
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(400, "Uploaded image is empty.")
    if CLIENT is None:
        raise HTTPException(
            501, "OCR not configured. Set GEMINI_API_KEY or disable this route.")
    try:
        extracted = ocr_image_to_text(img_bytes, mime)
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")
    tagged, matches = _detect(extracted or "")
    return JSONResponse(AnalyzeResponse(source="image", raw_text=extracted or "", tagged_text=tagged, matches=matches).model_dump())


@app.get("/patterns")
def get_patterns():
    return {"patterns": PATTERNS_RAW}


@app.post("/patterns")
def set_patterns(payload: dict = Body(...)):
    global PATTERNS_RAW, COMPILED
    items = payload.get("patterns", [])
    try:
        PATTERNS_RAW = _require_patterns(items)
        COMPILED = [{"label": p["label"], "pattern": p["pattern"],
                     "rx": build_pattern(p["pattern"])} for p in PATTERNS_RAW]
        return {"ok": True, "count": len(PATTERNS_RAW)}
    except Exception as e:
        raise HTTPException(400, str(e))


# =========================================================
# Entrypoint
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, reload=RELOAD, log_level=LOG_LEVEL)
