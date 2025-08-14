import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import regex as re
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Gemini SDK ---
from google import genai
from google.genai import types as gx

# =========================================================
# Env / Config
# =========================================================
load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Put it in your .env")

# Optional absolute/relative path to patterns json
PATTERNS_FILE_ENV = os.getenv("PATTERNS_FILE")

CLIENT = genai.Client(api_key=API_KEY)
app = FastAPI(title="ToxiScan", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Patterns: load, compile (mixed Sinhala + Latin, script ignored)
# =========================================================


def _require_patterns(items: Any) -> List[dict]:
    if not isinstance(items, list) or not items:
        raise RuntimeError(
            "No patterns found. Ensure your JSON has a non-empty 'patterns' list."
        )
    # Basic validation
    for i, p in enumerate(items):
        if not isinstance(p, dict):
            raise RuntimeError(
                f"Invalid pattern at index {i}: expected object.")
        if "label" not in p or "pattern" not in p:
            raise RuntimeError(
                f"Pattern at index {i} missing 'label' or 'pattern'.")
        if not isinstance(p["label"], str) or not isinstance(p["pattern"], str):
            raise RuntimeError(
                f"Pattern at index {i} has non-string 'label'/'pattern'.")
        if not p["pattern"].strip():
            raise RuntimeError(
                f"Pattern at index {i} has empty 'pattern' value.")
    return items


def load_patterns() -> List[dict]:
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
            print(f"[patterns] loaded {len(items)} from {p}")
            return items

    raise RuntimeError(
        "patterns.json not found. Set PATTERNS_FILE in .env or place patterns.json next to main.py."
    )


# l33t map for Latin; Sinhala handled literally (grapheme-safe)
LEET_MAP = {
    "a": "[a@]",
    "i": "[i1!|]",
    "o": "[o0]",
    "s": "[s$5]",
    "t": "[t7]",
    "e": "[e3]",
    "g": "[g9]",
    "b": "[b8]",
}
# up to 3 non-letter/number chars between letters
NOISE = r"(?:[^\p{L}\p{N}]{0,3})"


def is_latin_letter(ch: str) -> bool:
    o = ord(ch)
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z") or (0x00C0 <= o <= 0x024F)


def _latin_char_class(ch: str) -> str:
    low = ch.lower()
    return LEET_MAP.get(low, re.escape(ch))


def build_pattern(token: str) -> re.Pattern:
    """
    Mixed-script builder:
      - Latin letters: case-insensitive + l33t alternatives + allow repeats
      - Digits: allow repeats
      - Other unicode letters (e.g., Sinhala): literal + allow repeats
    Up to 3 non-alnum separators allowed between each step (NOISE).
    """
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


def compile_patterns(raw: List[dict]) -> List[dict]:
    # 'script' field (if present) is ignored
    return [
        {"label": p["label"], "pattern": p["pattern"],
            "rx": build_pattern(p["pattern"])}
        for p in raw
    ]


PATTERNS_RAW = load_patterns()
COMPILED = compile_patterns(PATTERNS_RAW)

# =========================================================
# Tagging utils
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


def find_matches(text: str):
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

# =========================================================
# OCR with Gemini (structured output)
# =========================================================


class OcrResult(BaseModel):
    text: str


def ocr_image_to_text(image_bytes: bytes, mime: str) -> str:
    resp = CLIENT.models.generate_content(
        model=MODEL_NAME,
        contents=[
            gx.Part.from_bytes(data=image_bytes, mime_type=mime),
            "Extract ALL text exactly as visible. Preserve newlines. Return JSON with a single field 'text'."
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": OcrResult,
            "temperature": 0
        },
    )
    if getattr(resp, "parsed", None) and getattr(resp.parsed, "text", None):
        return resp.parsed.text
    return json.loads(resp.text)["text"]

# =========================================================
# API Models
# =========================================================


class AnalyzeTextRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    source: str
    raw_text: str
    tagged_text: str
    matches: List[Dict[str, Any]]

# =========================================================
# Routes
# =========================================================


@app.post("/analyze/text", response_model=AnalyzeResponse)
async def analyze_text(req: AnalyzeTextRequest = Body(...)):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Provide non-empty 'text'.")
    tagged, matches = find_matches(req.text)
    return JSONResponse(AnalyzeResponse(
        source="text",
        raw_text=req.text,
        tagged_text=tagged,
        matches=matches
    ).model_dump())


@app.post("/analyze/image", response_model=AnalyzeResponse)
async def analyze_image(image: UploadFile = File(...)):
    mime = image.content_type or "image/jpeg"
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(400, "Uploaded image is empty.")
    try:
        extracted = ocr_image_to_text(img_bytes, mime)
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")
    tagged, matches = find_matches(extracted or "")
    return JSONResponse(AnalyzeResponse(
        source="image",
        raw_text=extracted or "",
        tagged_text=tagged,
        matches=matches
    ).model_dump())


@app.get("/patterns")
def get_patterns():
    return {"patterns": PATTERNS_RAW}


@app.post("/patterns")
def set_patterns(payload: dict = Body(...)):
    global PATTERNS_RAW, COMPILED
    items = payload.get("patterns", [])
    # Reuse the same validation
    PATTERNS_RAW = _require_patterns(items)
    COMPILED = compile_patterns(PATTERNS_RAW)
    return {"ok": True, "count": len(PATTERNS_RAW)}


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "patterns": len(COMPILED)}
