# ToxiScan – Sinhala/Singlish Profanity Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](#)

Lightweight, production-ready profanity detection for Sinhala, English, and Singlish.
Hybrid design: a tiny **AI word classifier** (fast, robust to obfuscations) with optional **regex rules** and **OCR** (image → text).

```
toxi_scan/
  server/         # FastAPI inference service (AI + optional rules + optional OCR)
  toxi_scan_ai/   # Model training project (data build + training + eval + export)
  toxiscan-app/   # Next.js frontend (demo UI with per-token confidence & threshold)
```

---

## ✨ Highlights

* **Speedy**: char n-gram TF-IDF + Logistic Regression (no GPU, ms/word)
* **Obfuscation-tolerant**: works with Singlish & creative spellings (e.g., `Hu—ttaaa`)
* **Threshold control**: pick operating point for precision/recall; UI slider included
* **Per-token confidence**: frontend shows probability for each detected token
* **OCR (optional)**: Gemini extracts text from images, then runs the same detector

---

## 🧱 Tech Stack

* **Model**: scikit-learn (`TfidfVectorizer(char 2–5) + LogisticRegression`)
* **Backend**: FastAPI + Uvicorn, optional Gemini OCR (`google.genai` or `google.generativeai`)
* **Frontend**: Next.js (App Router) + Tailwind + shadcn/ui
* **Artifacts**: `server/model/word_cuss_lr.joblib`, `server/model/threshold.txt`

---

## 📦 Installation

### 1) Model training deps

```bash
cd toxi_scan/toxi_scan_ai
pip install pandas scikit-learn joblib regex
# (optional, if you experiment with transformers later)
pip install torch transformers datasets
```

### 2) Server deps

```bash
cd ../server
pip install fastapi uvicorn regex python-dotenv joblib scikit-learn
# OCR (optional – either one works; both is fine)
pip install google-genai google-generativeai
```

### 3) Frontend deps

```bash
cd ../toxiscan-app
npm install
```

---

## 📚 Data & Training (in `toxi_scan_ai/`)

### Inputs

* `normal.csv` with columns: **Sinhala**, **English**, **Singlish**
  (We extract words from Sinhala + Singlish; English is ignored for the word model.)
* `cuss.json` with:

  ```json
  { "patterns": [ { "label": "profanity-singlish", "pattern": "Hutta" }, ... ] }
  ```

### Build dataset → `data/words.csv`

* Unicode NFKC → lowercase
* Strip zero-width chars `[\u200B-\u200D\uFEFF]`
* Compress long repeats (`aaaaa → aa`)
* Tokenization: `\b[\p{L}\p{N}][\p{L}\p{N}\-._\u200B-\u200D]*\b`

(Use the provided `create_dataset.py` from the training folder.)

### Train the model

* Vectorizer: `TfidfVectorizer(analyzer="char", ngram_range=(2,5))`
* Classifier: `LogisticRegression(max_iter=2000, class_weight="balanced")`
* Stratified split, then **threshold search** to maximize F1 on "cuss"
* Exports:

  ```
  toxi_scan_ai/models/word_cuss_lr.joblib
  toxi_scan_ai/models/threshold.txt
  ```

Copy these to the server:

```
toxi_scan/server/model/word_cuss_lr.joblib
toxi_scan/server/model/threshold.txt
```

### Example validation (from a real run)

* Best operating threshold **τ ≈ 0.746**
* **AUPRC (cuss)** ≈ **0.4767**
* At τ=0.5 (not recommended): cuss P=0.375, R=0.656, F1=0.477
  (we deploy with the learned τ for higher precision)

---

## 🔌 Server (FastAPI) – `toxi_scan/server`

### Env (.env)

```env
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
DETECTOR=ai             # ai | rules | hybrid

# OCR (optional) – either var name works
GEMINI_API_KEY=your_key   # or GOOGLE_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash
```

### Run

```bash
cd toxi_scan/server
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Endpoints

#### `GET /health`

Returns service status:

```json
{
  "ok": true,
  "detector": "ai",
  "ai_loaded": true,
  "ai_threshold": 0.746,
  "patterns": 0,
  "ocr_enabled": true,
  "ocr_sdk": "google.genai",
  "gemini_model": "gemini-2.0-flash",
  "api_key_present": true
}
```

#### `GET /is_cuss?word=...&threshold?=0.8`

Quick word check:

```json
{ "word": "Hutta", "normalized": "hutta", "is_cuss": true, "score": 0.966, "threshold": 0.746 }
```

#### `POST /analyze/text`

```json
{ "text": "හෙලෝ Hutta 😅 Hu—ttaaa සුභ!", "threshold": 0.75 }
```

Response:

```json
{
  "source": "text",
  "raw_text": "…",
  "tagged_text": "හෙලෝ <cuss>Hutta</cuss> 😅 <cuss>Hu—ttaaa</cuss> සුභ!",
  "matches": [
    { "label": "ai-cuss", "match": "Hutta", "normalized": "hutta", "score": 0.9659, "start": 6, "end": 11 },
    { "label": "ai-cuss", "match": "Hu—ttaaa", "normalized": "hu—ttaa", "score": 0.9642, "start": 15, "end": 23 }
  ]
}
```

#### `POST /analyze/image`  *(optional OCR)*

Form-data: `image=@file.jpg`
Extracts text via Gemini then runs the same token-level detection.

#### (optional) Patterns endpoints

* `GET /patterns` → returns current pattern list (if present)
* `POST /patterns` → replace pattern list (used in `rules` / `hybrid` modes)

---

## 🖥 Frontend (Next.js) – `toxiscan-app`

### Config

```env
# toxiscan-app/.env.local
NEXT_PUBLIC_TOXISCAN_API=http://localhost:8000
```

### Run

```bash
cd toxi_scan/toxiscan-app
npm run dev
```

### What you get

* Text/Image tabs (image uses server OCR if enabled)
* **Threshold slider** (0.30–0.95) with default fetched from `/health`
* Inline highlighting via `<cuss>…</cuss>`
* **Per-token confidence table** and **overall confidence** (avg of token scores)
* Copy/clear actions, shadcn styling

---

## 🔁 Typical Flow

```
User Text/Image
   ↓
Frontend → POST /analyze/text or /analyze/image
   ↓
Server:
  - (image) OCR via Gemini → text
  - tokenize → normalize → predict_proba(word)
  - compare to threshold τ
  - merge spans → <cuss>…</cuss>
  - return matches + scores
   ↓
Frontend renders highlights + per-token & overall confidence
```

---

## ⚖ Operating Point & Accuracy

* Model is trained on imbalanced data; we **learn the best threshold** on validation to favor high precision.
* Example: **τ ≈ 0.746**, **AUPRC ≈ 0.477** on the cuss class.
* The UI exposes the threshold so teams can dial precision/recall to taste.

---

## 📁 Artifacts & Versioning

* Model file: `server/model/word_cuss_lr.joblib`
* Threshold file: `server/model/threshold.txt`
* `/health` exposes the active threshold; replace artifacts to upgrade the model.

---

## 👥 Team & Responsibilities

* **21UG1056** – Data Engineering & Curation
* **21UG1287** – Data Engineering & Curation
* **21UG1073** – Data Engineering & Curation, Model Training, QA & Documentation
* **21UG1376** – Data Engineering & Curation, Backend API & OCR, Frontend
* **21UG1091** – Model Training
* **21UG1092** – Model Training
* **21UG1149** – Backend API & OCR
* **21UG0460** – Backend API & OCR
* **21UG951** – Frontend
* **21UG1079** – QA & Documentation
* **21UG1260** – QA & Documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⭐ Support

If you find this project helpful, please give it a star!
