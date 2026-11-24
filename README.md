# 🔎 Agentic Fact-Checking System (FastAPI Edition)

An enterprise-ready agentic AI pipeline that verifies claims using real-time web retrieval, dense vector similarity, cross-encoder neural reranking, and Google Gemini reasoning.

---

## 🛠️ System Architecture

1. **Web Search**: Real-time querying via DuckDuckGo Search (`ddgs`) to collect up-to-date sources.
2. **Dense Semantic Retrieval**: Vector embeddings generated with `sentence-transformers/all-MiniLM-L6-v2` to select candidate documents.
3. **Cross-Encoder Reranking**: Re-evaluates query-document pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2` for precise relevance scoring.
4. **LLM Reasoning**: Google Gemini (`gemini-2.5-flash` / `gemini-flash-latest`) synthesizes evidence to produce a verdict (`SUPPORTED`, `REFUTED`, `UNCERTAIN`), confidence score, and citations.

---

## 🚀 Getting Started

### 1. Environment Setup

Ensure your `.env` file contains your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
LLM_MODEL=gemini-2.5-flash
```

Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### 2. Launch the FastAPI Server & Dashboard (Recommended)

Run the FastAPI server with Uvicorn:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### 3. API Endpoints

#### `POST /api/verify`
Fact-checks a given claim.

**Request Body:**
```json
{
  "claim": "Australia won the 2023 ICC Men's Cricket World Cup.",
  "top_k_retrieval": 10,
  "top_k_rerank": 5
}
```

**Response Example:**
```json
{
  "claim": "Australia won the 2023 ICC Men's Cricket World Cup.",
  "verdict": "SUPPORTED",
  "confidence": 100,
  "explanation": "Australia defeated India in the 2023 Cricket World Cup final at Narendra Modi Stadium...",
  "citations": [
    "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup_Final"
  ],
  "evidence": [
    {
      "title": "2023 Cricket World Cup final - Wikipedia",
      "url": "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup_Final",
      "score": 0.9412,
      "content": "..."
    }
  ],
  "execution_time_seconds": 1.45
}
```

#### `GET /api/health`
Checks system status and preloaded models.

#### `GET /api/examples`
Fetches curated test claims.

---

### 4. Run via CLI Pipeline

Verify claims directly from the terminal:

```powershell
.\venv\Scripts\python.exe src\pipeline.py
```
