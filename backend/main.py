"""
Codebase RAG Assistant — Backend
FastAPI + ChromaDB + OpenAI
"""

import os
from dotenv import load_dotenv
load_dotenv("../.env")
import ast
import re
import shutil
import tempfile
import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import threading
import git
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import uvicorn

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(title="Codebase RAG Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients ─────────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path="./chroma_db")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Constants ───────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sh": "bash",
    ".md": "markdown",
}

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", "vendor", "target",
}

IGNORE_FILES = {
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
    "Cargo.lock", "go.sum",
}

MAX_FILE_SIZE = 80_000   # chars — skip minified/generated files
CHUNK_BATCH  = 500       # docs to upsert at once

# ── Pydantic models ─────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source: str              # GitHub URL or local path
    repo_name: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    repo_name: str
    top_k: int = 6

# ── Helpers ─────────────────────────────────────────────────────────────────

def safe_collection_name(name: str) -> str:
    """ChromaDB collection names must be alphanumeric + underscores."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name[0].isdigit():
        name = "r_" + name
    return name[:60]


def get_or_create_collection(repo_name: str):
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    return chroma_client.get_or_create_collection(
        name=safe_collection_name(repo_name),
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

# ── Code parsers ─────────────────────────────────────────────────────────────

def chunk_by_lines(content: str, chunk_size: int = 60, overlap: int = 10) -> List[dict]:
    """Fallback: split file into overlapping line windows."""
    lines = content.split("\n")
    chunks, i = [], 0
    while i < len(lines):
        block = "\n".join(lines[i : i + chunk_size])
        if block.strip():
            chunks.append({
                "code": block,
                "function_name": f"lines_{i+1}_to_{min(i+chunk_size, len(lines))}",
                "start_line": i + 1,
                "end_line": min(i + chunk_size, len(lines)),
                "type": "chunk",
            })
        i += chunk_size - overlap
    return chunks


def parse_python(content: str) -> List[dict]:
    """AST-based chunking for Python — extracts top-level functions & classes."""
    chunks = []
    try:
        tree = ast.parse(content)
        lines = content.split("\n")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Skip heavily nested nodes (keep methods inside classes, skip inner funcs)
                code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                if len(code.strip()) < 10:
                    continue
                chunks.append({
                    "code": code,
                    "function_name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                })
    except SyntaxError:
        return chunk_by_lines(content)
    return chunks or chunk_by_lines(content)


def parse_js_ts(content: str) -> List[dict]:
    """Regex-based function boundary detection for JS/TS."""
    FUNC_RE = re.compile(
        r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)|"
        r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\(|function)|"
        r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)|"
        r"^\s{2,4}(?:public|private|protected|static|async|\s)*(\w+)\s*\(",
        re.MULTILINE,
    )
    lines = content.split("\n")
    starts = []
    for i, line in enumerate(lines):
        m = FUNC_RE.match(line)
        if m:
            name = next((g for g in m.groups() if g), "anonymous")
            starts.append((i, name))

    if not starts:
        return chunk_by_lines(content)

    chunks = []
    for idx, (start, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        code = "\n".join(lines[start:end])
        if len(code.strip()) < 10:
            continue
        chunks.append({
            "code": code,
            "function_name": name,
            "start_line": start + 1,
            "end_line": end,
            "type": "function",
        })
    return chunks or chunk_by_lines(content)


def parse_generic(content: str) -> List[dict]:
    """Language-agnostic chunking by brace-depth heuristic, falling back to line windows."""
    # Try to find top-level blocks by { ... } depth
    lines = content.split("\n")
    starts, depth = [], 0
    for i, line in enumerate(lines):
        opens  = line.count("{")
        closes = line.count("}")
        if depth == 0 and opens > 0:
            starts.append(i)
        depth = max(0, depth + opens - closes)

    if len(starts) < 2:
        return chunk_by_lines(content)

    chunks = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        code = "\n".join(lines[start:end])
        if len(code.strip()) < 10:
            continue
        chunks.append({
            "code": code,
            "function_name": f"block_{start+1}",
            "start_line": start + 1,
            "end_line": end,
            "type": "block",
        })
    return chunks or chunk_by_lines(content)


def parse_file(content: str, language: str) -> List[dict]:
    if language == "python":
        return parse_python(content)
    elif language in ("javascript", "typescript"):
        return parse_js_ts(content)
    elif language in ("go", "java", "cpp", "c", "rust", "csharp", "kotlin", "swift", "php"):
        return parse_generic(content)
    else:
        return chunk_by_lines(content)

# ── Ingestion ────────────────────────────────────────────────────────────────

def ingest_repo(source: str, repo_name: str) -> dict:
    temp_dir = None
    try:
        if source.startswith("http") or source.startswith("git@"):
            temp_dir = tempfile.mkdtemp()
            print(f"[ingest] Cloning {source} ...")
            git.Repo.clone_from(source, temp_dir, depth=1)
            repo_path = Path(temp_dir)
        else:
            repo_path = Path(source).expanduser()
            if not repo_path.exists():
                raise ValueError(f"Path '{source}' does not exist")

        collection = get_or_create_collection(repo_name)

        documents: List[str] = []
        metadatas: List[dict] = []
        ids: List[str]       = []
        doc_id = 0
        files_processed = 0

        for file_path in sorted(repo_path.rglob("*")):
            if any(ignored in file_path.parts for ignored in IGNORE_DIRS):
                continue
            if not file_path.is_file():
                continue
            if file_path.name in IGNORE_FILES:
                continue

            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            language = SUPPORTED_EXTENSIONS[ext]
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if not content.strip() or len(content) > MAX_FILE_SIZE:
                continue

            relative_path = str(file_path.relative_to(repo_path))
            chunks = parse_file(content, language)
            files_processed += 1

            for chunk in chunks:
                text = (
                    f"File: {relative_path}\n"
                    f"Function/Block: {chunk['function_name']}\n"
                    f"Lines: {chunk['start_line']}-{chunk['end_line']}\n\n"
                    f"{chunk['code']}"
                )
                documents.append(text)
                metadatas.append({
                    "file_path": relative_path,
                    "function_name": chunk["function_name"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "language": language,
                    "type": chunk["type"],
                    "repo": repo_name,
                })
                ids.append(f"{safe_collection_name(repo_name)}_{doc_id}")
                doc_id += 1

                if len(documents) >= CHUNK_BATCH:
                    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                    documents, metadatas, ids = [], [], []
                    print(f"[ingest] Indexed {doc_id} chunks so far...")

        if documents:
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

        print(f"[ingest] Done — {doc_id} chunks from {files_processed} files")
        return {"chunks": doc_id, "files": files_processed}

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    key = os.getenv("OPENAI_API_KEY", "")
    return {
        "status": "ok",
        "openai_key_configured": bool(key and key != "your_openai_api_key_here"),
    }


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    source = req.source.strip()

    if not req.repo_name:
        if source.startswith("http") or source.startswith("git@"):
            repo_name = source.rstrip("/").split("/")[-1].replace(".git", "")
        else:
            repo_name = Path(source).name
    else:
        repo_name = req.repo_name

    try:
        stats = ingest_repo(source, repo_name)
        return {
            "status": "success",
            "repo_name": repo_name,
            "chunks_indexed": stats["chunks"],
            "files_processed": stats["files"],
        }
    except Exception as e:
        print(f"[ingest error] {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/repos")
def list_repos():
    cols = chroma_client.list_collections()
    repos = []
    for col in cols:
        c = chroma_client.get_collection(col.name)
        repos.append({"name": col.name, "chunk_count": c.count()})
    return {"repos": repos}


@app.delete("/repos/{repo_name}")
def delete_repo(repo_name: str):
    try:
        chroma_client.delete_collection(safe_collection_name(repo_name))
        return {"status": "deleted", "repo": repo_name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/query")
def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        collection = get_or_create_collection(req.repo_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Repo not found: {e}")

    total = collection.count()
    if total == 0:
        raise HTTPException(status_code=404, detail="Repo has no indexed chunks")

    results = collection.query(
        query_texts=[req.question],
        n_results=min(req.top_k, total),
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant code found")

    # Build context block
    context_parts = []
    sources = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        context_parts.append(f"[Source {i+1}]\n{doc}")
        sources.append({
            "index": i + 1,
            "file": meta["file_path"],
            "function": meta["function_name"],
            "lines": f"{meta['start_line']}–{meta['end_line']}",
            "language": meta["language"],
            "similarity": round(1 - dist, 3),
            "code": doc.split("\n\n", 1)[1] if "\n\n" in doc else doc,
        })

    context = "\n\n" + "─" * 60 + "\n\n".join(context_parts)

    system = (
        "You are a senior software engineer with deep expertise in reading and explaining codebases. "
        "When answering, be specific: name functions, reference line numbers, explain logic step by step. "
        "Use markdown with fenced code blocks. At the end, summarise which sources were most relevant."
    )
    user_msg = (
        f"I have a question about this codebase:\n\n**{req.question}**\n\n"
        f"Here are the most relevant code sections retrieved from the vector store:\n\n{context}\n\n"
        "Please provide a thorough, accurate answer using only the code above."
    )

    def stream():
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
                stream=True,
                temperature=0.1,
                max_tokens=2048,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
