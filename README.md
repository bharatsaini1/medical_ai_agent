# 🧬 Medical AI Agent — Hybrid RAG + Web Search

A production-ready medical assistant powered by **Llama 3 (Groq)**, **Hybrid RAG** (BM25 + Pinecone), and **Tavily Web Search** fallback.

---

## 🏗️ Architecture

```
User Query
    ↓
[1] Query Rewriting (LLM improves retrieval quality)
    ↓
[2] Hybrid Retrieval
    ├── BM25 (keyword exact match)
    └── Pinecone Vector Search (semantic similarity)
    ↓
[3] Confidence Check (threshold = 0.35)
    ├── High confidence → RAG context → LLM
    └── Low confidence  → Tavily Web Search → LLM
    ↓
[4] Llama 3 via Groq generates grounded answer
    ↓
[5] Response + Confidence Score + Medical Disclaimer
```

---

## 📁 Project Structure

```
medical_ai_agent/
├── data/
│   └── Diseases_Symptoms.csv       ← your 405-disease dataset
│
├── embeddings/
│   ├── documents.json              ← cached processed docs (auto-generated)
│
├── retriever/
│   ├── bm25.py                     ← BM25 keyword retriever
│   ├── vector.py                   ← Pinecone vector retriever
│   └── hybrid.py                   ← combined hybrid retriever
│
├── agent/
│   └── agent.py                    ← LangGraph AI agent
│
├── tools/
│   └── web_search.py               ← Tavily web search tool
│
├── api/
│   └── app.py                      ← Flask REST API
│
├── utils/
│   ├── config.py                   ← centralized configuration
│   ├── logger.py                   ← loguru logging setup
│   └── data_processor.py           ← dataset cleaning & prep
│
├── scripts/
│   ├── setup.py                    ← one-time setup (run first!)
│   └── test_queries.py             ← automated test runner
│
├── frontend/
│   └── index.html                  ← production UI (open in browser)
│
├── logs/                           ← auto-created log files
├── .env.example                    ← copy this to .env
└── requirements.txt
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone / create project directory

```bash
cd medical_ai_agent
```

### Step 2 — Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Key | Where to get it |
|-----|----------------|
| `GROQ_API_KEY` | https://console.groq.com (free) |
| `PINECONE_API_KEY` | https://app.pinecone.io (free tier available) |
| `TAVILY_API_KEY` | https://tavily.com (free tier available) |

### Step 5 — Run one-time setup

```bash
python scripts/setup.py
```

This will:
- Validate API keys
- Clean and process your 405-disease dataset
- Generate embeddings and upload to Pinecone (~2 min first run)

### Step 6 — Start the server

```bash
python api/app.py
```

Server runs at: `http://localhost:5000`

### Step 7 — Open the frontend

Open `frontend/index.html` in your browser. That's it! 🎉

---

## 🔌 API Reference

### `POST /chat`

```json
{
  "query": "I have intense itching at night with small blisters",
  "session_id": "optional_user_id"
}
```

**Response:**
```json
{
  "response":        "Based on your symptoms...\n\n📊 Confidence Score: 82% | Source: Medical Database",
  "confidence":      0.82,
  "confidence_pct":  82,
  "source":          "database",
  "used_web_search": false,
  "top_matches": [
    {"name": "Scabies", "score": 0.91},
    {"name": "Eczema",  "score": 0.74}
  ],
  "session_id":      "abc123",
  "processing_time": 1.43
}
```

### `GET /health`

Returns server status, model name, and dataset size.

### `POST /reset`

```json
{ "session_id": "abc123" }
```

Clears chat history for the given session.

---

## 🧪 Test Queries

```bash
# Server must be running first
python scripts/test_queries.py
```

Sample queries to try:
- `"Intense itching especially at night, small blisters on hands"` → Scabies
- `"Cloudy eyes, excessive tearing in my child"` → Congenital Glaucoma
- `"Persistent fatigue, weight gain, always cold, dry skin"` → Hypothyroidism
- `"Sad all the time, lost interest in everything"` → Depression
- `"Severe heartburn, acid coming up throat after eating"` → GERD

---

## 🐛 Debug Tips

### Server won't start
- Check `.env` has all 3 API keys filled in
- Run `python scripts/setup.py` first

### Pinecone error: index not found
- Run `python scripts/setup.py` — it creates the index
- Check your Pinecone dashboard for the index name

### Low confidence / always using web search
- Lower `CONFIDENCE_THRESHOLD` in `.env` (try `0.25`)
- Ensure documents were indexed: check Pinecone dashboard

### Groq rate limit errors
- The free tier allows ~30 requests/minute
- Add a delay between requests in test_queries.py

### Embeddings mismatch error
- Delete `embeddings/` folder and run setup again

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Groq LLM API key |
| `PINECONE_API_KEY` | required | Pinecone vector DB key |
| `TAVILY_API_KEY` | required | Tavily web search key |
| `PINECONE_INDEX_NAME` | `medical-rag-index` | Pinecone index name |
| `TOP_K` | `5` | Number of RAG results |
| `CONFIDENCE_THRESHOLD` | `0.35` | Min score before web fallback |
| `BM25_WEIGHT` | `0.5` | BM25 blend weight |
| `VECTOR_WEIGHT` | `0.5` | Vector blend weight |
| `FLASK_PORT` | `5000` | API server port |

---

## ⚠️ Medical Disclaimer

This system is for **educational purposes only**. It does not provide medical diagnosis or treatment advice. Always consult a qualified healthcare professional.
