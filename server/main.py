import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

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

# optional absolute/relative path
PATTERNS_FILE_ENV = os.getenv("PATTERNS_FILE")
AUTO_CREATE = os.getenv("CREATE_PATTERNS", "1") == "1"

CLIENT = genai.Client(api_key=API_KEY)

app = FastAPI(title="ToxiScan", version="1.1")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Patterns: load, compile (Sinhala + Singlish/Latin)
# =========================================================


def load_patterns() -> List[dict]:
    defaults = [{"label": "huth-stem", "pattern": "huth", "script": "latin"},
                {"label": "whut-stem", "pattern": "whut", "script": "latin"}]
    candidates: List[Path] = []
    if PATTERNS_FILE_ENV:
        candidates.append(Path(PATTERNS_FILE_ENV))
    candidates.append(Path.cwd() / "patterns.json")
    candidates.append(Path(__file__).parent / "patterns.json")

    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("patterns", [])
            if items:
                print(f"[patterns] loaded {len(items)} from {p}")
                return items
            else:
                print(
                    f"[patterns] {p} has no 'patterns' entries, using defaults.")
    if AUTO_CREATE:
        skeleton_path = Path.cwd() / "patterns.json"
        skeleton = {"patterns": defaults}
        skeleton_path.write_text(json.dumps(
            skeleton, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[patterns] created skeleton at {skeleton_path}")
    print("[patterns] using in-code defaults")
    return defaults


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


def _latin_char_class(ch: str) -> str:
    low = ch.lower()
    return LEET_MAP.get(low, re.escape(ch))


def build_pattern(token: str, script: str) -> re.Pattern:
    parts = []
    if script.lower() == "latin":
        for ch in token:
            parts.append(f"(?:{_latin_char_class(ch)})+")
        flags = re.IGNORECASE
    else:
        # Sinhala/other scripts: literal graphemes, allow repeats
        for ch in token:
            parts.append(f"(?:{re.escape(ch)})+")
        flags = 0
    body = NOISE.join(parts)
    return re.compile(body, flags | re.UNICODE)


def compile_patterns(raw: List[dict]) -> List[dict]:
    return [
        {"label": p["label"], "pattern": p["pattern"],
            "rx": build_pattern(p["pattern"], p.get("script", "latin"))}
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
            ls, le, llbl = merged[-1]
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
            info.append({"label": label, "match": text[m.start(
            ):m.end()], "start": m.start(), "end": m.end()})
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
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "Provide non-empty 'patterns' list.")
    PATTERNS_RAW = items
    COMPILED = compile_patterns(PATTERNS_RAW)
    return {"ok": True, "count": len(PATTERNS_RAW)}


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "patterns": len(COMPILED)}
