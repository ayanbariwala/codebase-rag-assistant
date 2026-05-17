# ⚡ Codebase RAG Assistant

> Ask questions about any GitHub repository in plain English. The assistant retrieves the most relevant code chunks using vector similarity search and answers with GPT-4o — with exact source file citations.

![Demo](docs/screenshot.png)

![Architecture](./docs/architecture.png)

## 🏗️ Architecture

```
Phase 1 — Ingestion
─────────────────────────────────────────────────────────────
GitHub Repo ──clone──► Code Parser ──chunks──► Embeddings ──► ChromaDB
                       (AST / regex)           (text-embedding-3-small)   (vector store)

Phase 2 — Query
─────────────────────────────────────────────────────────────
User Question ──embed──► ChromaDB ──top-k──► GPT-4o ──► Answer + Sources
                         (cosine similarity)
```

**Key design decisions:**
- **AST-based chunking** for Python (extracts functions & classes); regex-based for JS/TS; line-window fallback for everything else
- **ChromaDB** runs fully locally — no external vector DB needed, all data persists on disk
- **Streaming SSE** responses so answers appear token-by-token, just like ChatGPT
- **Source citations** show exact file, function name, and line numbers for every answer

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| Vector Store | ChromaDB (local, persistent) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o` |
| Code Parsing | Python `ast` module + regex |
| Git cloning | GitPython |
| Frontend | Vanilla JS + Highlight.js + Marked.js |

## ⚙️ Setup

### Prerequisites
- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/codebase-rag-assistant
cd codebase-rag-assistant
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 3. Install backend dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start the backend

```bash
# From the backend/ directory, with venv active:
python main.py
# → Server running at http://localhost:8000
```

### 5. Open the frontend

```bash
# From the project root, open a new terminal:
cd frontend
python -m http.server 3000
# → Open http://localhost:3000 in your browser
```

That's it! No Docker, no databases to configure, no build steps.

## 🚀 Usage

1. **Paste a GitHub URL** into the sidebar (e.g. `https://github.com/tiangolo/fastapi`)
2. Click **Index Repository** — the repo is cloned, parsed, embedded, and stored locally
3. **Select the repo** from the sidebar list
4. **Ask questions** like:
   - *"How does authentication work?"*
   - *"Walk me through the request lifecycle"*
   - *"What does the `create_user` function do?"*
5. See the streaming answer with **source code citations** — click any source to expand the code

## 📁 Project Structure

```
codebase-rag-assistant/
├── backend/
│   ├── main.py          ← FastAPI app, all endpoints + RAG pipeline
│   └── requirements.txt
├── frontend/
│   └── index.html       ← Single-file UI (no build step needed)
├── chroma_db/           ← Created automatically, local vector store
├── .env.example
└── README.md
```

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check server + API key status |
| `/ingest` | POST | Clone & index a repo |
| `/repos` | GET | List all indexed repos |
| `/repos/{name}` | DELETE | Remove a repo from the index |
| `/query` | POST | Stream an answer (SSE) |

### Example: Ingest a repo

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "https://github.com/tiangolo/fastapi"}'
```

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does dependency injection work?", "repo_name": "fastapi"}'
```


## 📊 Supported Languages

Python · JavaScript · TypeScript · Go · Java · Rust · C/C++ · C# · Ruby · PHP · Kotlin · Swift · Bash · Markdown


