# ToxiScan

A full-stack app for detecting offensive language and hate speech in Sinhala, English, and Singlish text or images.
The system combines rule-based pattern matching with Google Gemini OCR for image analysis.

---

## Project Structure

| Path | Description |
|------|-------------|
| `server/` | FastAPI backend that exposes text/image analysis endpoints and handles pattern matching. |
| `server/patterns.json` | Default list of offensive words/phrases. |
| `toxiscan-app/` | Next.js frontend for submitting text or images and viewing results. |

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for Next.js 15)
- A Google **Gemini API key**  
  (used by the backend for OCR).

---

## Backend Setup (`server/`)

1. **Create and activate a virtual environment**

   ```bash
   cd server
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment variables**

   Create a `.env` file with at least:

   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.0-flash   # optional override
   PATTERNS_FILE=path/to/patterns.json  # optional
   ```

4. **Run the API**

   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Available Endpoints

| Method & Route | Description |
|----------------|-------------|
| `POST /analyze/text` | Analyze raw text. Body: `{ "text": "..." }` |
| `POST /analyze/image` | Analyze an image. Multipart form with field `image`. |
| `GET /patterns` | Retrieve current pattern list. |
| `POST /patterns` | Replace pattern list. Body: `{ "patterns": [...] }` |
| `GET /health` | Service health check. |

Responses include the original text, a version with `<cuss>...</cuss>` tags, and metadata about each match.

---

## Frontend Setup (`toxiscan-app/`)

1. **Install dependencies**

   ```bash
   cd toxiscan-app
   npm install
   # or: pnpm install / yarn install
   ```

2. **Configure API endpoint**

   Create `.env.local`:

   ```
   NEXT_PUBLIC_TOXISCAN_API=http://localhost:8000
   ```

3. **Run the development server**

   ```bash
   npm run dev
   ```

   Visit <http://localhost:3000> to use the UI.

---

## Using the Application

1. Navigate to the web UI.
2. Choose **Text** or **Image** mode.
3. Enter text or upload an image.
4. Click **Analyze** to view tagged results and the number of detected terms.

---

## Customizing Patterns

- Edit `server/patterns.json` or use the `POST /patterns` endpoint to supply a new list.
- Each entry requires `label` and `pattern` fields (strings).  
  Patterns are converted into robust regex rules that handle case, spacing, leet characters, and mixed Sinhala/Latin scripts.

---

## License

No license is specified in the repository. If you plan to publish or distribute this project, consider adding an appropriate license file.

---

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) for backend framework.
- [Next.js](https://nextjs.org/) & [Tailwind CSS](https://tailwindcss.com/) for frontend.
- Google [Gemini API](https://ai.google.dev/) for OCR capability.

Feel free to copy this README into your repository and adapt it to fit any additional details or deployment instructions specific to your setup.

