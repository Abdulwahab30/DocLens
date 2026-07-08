<div align="center">

# 🔍 Doc Lens

### *Chat with your documents. Get answers grounded in your PDFs — with exact page citations.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC143C?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM_API-6C47FF?style=for-the-badge)](https://openrouter.ai)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

Upload PDFs → Ask questions → Get answers grounded in your documents,  
with **page citations**, **table awareness**, and **figure descriptions** powered by vision AI.

</div>

---

## ✨ What Doc Lens Does

You upload one or more PDF files. Behind the scenes the system:

| Step | What happens |
|------|-------------|
| 📄 **Parse** | Reads every page — text, tables, and figures — using Docling |
| 🏷️ **Chunk** | Splits content into page-bounded segments with bounding-box provenance |
| 👁️ **Enrich** | Sends figures to a vision LLM to generate search-friendly descriptions |
| 🧠 **Embed** | Converts every chunk into a 384-dim semantic vector stored in Qdrant |
| 🔍 **Retrieve** | Hybrid search (vector + keyword) → RRF merge → cross-encoder rerank |
| 💬 **Answer** | Grounds the LLM reply in the top-5 chunks and cites the exact page |

> This is **Retrieval-Augmented Generation (RAG)** — the AI reads your documents first, then answers. No hallucination. No guessing from stale training data.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    Browser["🌐 Browser\nLogin · Chat · Upload PDF"]

    Browser -->|"POST /api/documents"| FastAPI
    Browser -->|"POST /api/chat"| FastAPI

    FastAPI["⚡ FastAPI\nJWT auth · route guards · async handlers"]

    FastAPI -->|"save metadata"| Postgres[("🐘 PostgreSQL\nusers · documents\nchunks · chat logs")]
    FastAPI -->|"store raw file"| MinIO[("🗄️ MinIO\nS3-compatible\nobject storage")]
    FastAPI -->|"queue task"| Redis[("🔴 Redis\ntask broker")]

    Redis -->|"pick up task"| Celery

    Celery["⚙️ Celery Worker\n1 Download PDF from MinIO\n2 Docling parses PDF\n3 OCR fallback if needed\n4 Vision LLM for figures\n5 Embed chunks\n6 Upsert to Qdrant\n7 Mark status = ready"]

    Celery -->|"write chunks"| Postgres
    Celery -->|"upsert vectors"| Qdrant[("🔷 Qdrant\nvector database")]

    FastAPI -->|"embed question"| Embed["🤖 sentence-transformers\nBAAI/bge-small-en-v1.5"]
    Embed -->|"top-30 by cosine sim"| Qdrant
    FastAPI -->|"top-20 keyword match"| Postgres
    Qdrant & Postgres -->|"RRF merge"| Reranker["📊 Cross-Encoder\nms-marco-MiniLM-L-6-v2\ntop-5 chunks"]
    Reranker -->|"grounded prompt"| LLM["🌟 OpenRouter LLM\nGrounded answer + citations"]
    LLM --> Browser
```

---

## 📊 Evaluation

Tested on a 20-question benchmark across 2 documents (a renewable-energy report and a space-exploration guide), using `eval/run_eval.py`.

| Metric | Score |
|--------|-------|
| Retrieval hit-rate@5 | **100%** (20/20) |
| Faithfulness (LLM judge) | Not measured this run — see note below |

*Methodology: Questions were written from the source documents with known correct pages. Hit-rate measures whether the true source page appeared in the top-5 retrieved chunks (`retrieve_relevant_chunks` — dense vector search + Postgres full-text search, merged with RRF, then cross-encoder reranked). Faithfulness is graded by an LLM judge (`eval/run_eval.py --judge`) that is shown the actual retrieved chunk text and the assistant's generated answer, and asked whether the answer only uses information present in that context (`FAITHFUL`/`UNFAITHFUL` + a one-sentence rationale).*

**Note on this run:** hit-rate (100%, 20/20) comes from a clean API-mode pass (`--mode api`, the default — exercises the full stack including the LLM-generated reply) against a freshly-uploaded copy of both eval PDFs. While re-running with `--judge` to get a faithfulness score, a real bug surfaced and was fixed: `fetch_chunk_texts()` in `eval/run_eval.py` reused one pooled DB connection across multiple `asyncio.run()` calls, each with its own event loop, causing `another operation is in progress` errors — fixed by disposing the engine after each fetch. With that fixed, the judge calls (and, shortly after, the `/api/chat` calls themselves) started failing with `429 Too Many Requests` — the project's free-tier OpenRouter key (`free-models-per-day`, 50 req/day) ran out of quota mid-evaluation-session, confirmed in the server logs. Re-run `python eval/run_eval.py --email <you> --password <pw> --judge` once the quota resets (or after adding OpenRouter credits) to get a faithfulness number from this exact pipeline.

---

## 🧠 How the Retrieval Pipeline Works

```
User question
     │
     ├──► Embed question  ──────────────────► Qdrant vector search  ──► top-30 IDs
     │                                                                        │
     └──► Postgres full-text search (FTS)  ──────────────────────────► top-20 IDs
                                                                              │
                                                             Reciprocal Rank Fusion (RRF)
                                                                              │
                                                             Cross-encoder reranker
                                                                              │
                                                                        top-5 chunks
                                                                              │
                                                             OpenRouter LLM (grounded)
                                                                              │
                                                                  Answer + citations
```

---

## 📚 Concepts Explained

<details>
<summary><b>🤔 What is RAG?</b></summary>

A normal AI chatbot answers from its training data — whatever it learned before it was released. **RAG** adds a retrieval step: before answering, the system searches your documents for the most relevant passages and hands them to the AI as extra context. The AI then answers based on what your documents actually say, and cites the exact page it used. This prevents hallucination and keeps answers grounded in your data.

</details>

<details>
<summary><b>🧮 Embeddings & Vector Search</b></summary>

An "embedding" is a list of numbers that represents the *meaning* of a sentence. Sentences with similar meaning get similar numbers, even if they use different words. For example, `"heart attack"` and `"myocardial infarction"` end up with nearly identical embedding vectors.

This project uses **`BAAI/bge-small-en-v1.5`** — a 130 MB model that runs locally with no API key. It converts every text chunk into a **384-dimensional vector**.

When you ask a question, the question is also embedded. The system finds the chunks whose vectors are closest to the question vector — the most semantically relevant passages.

</details>

<details>
<summary><b>🔷 Qdrant — Vector Database</b></summary>

Qdrant is a database designed for storing and searching embedding vectors. Think of it as a database index optimised for *"find the nearest neighbours in high-dimensional space"*.

Each entry stores the vector plus a small payload (`user_id`, `document_id`, `chunk_id`). The full chunk text stays in Postgres. Every search is filtered by `user_id` so users cannot see each other's documents.

</details>

<details>
<summary><b>🔀 Hybrid Search & Reciprocal Rank Fusion (RRF)</b></summary>

Semantic vector search is great for concepts but can miss exact keywords. If your document mentions a product code like `"ALPHA-7734"`, it might not be close enough in vector space to match.

Postgres has a built-in full-text search engine (`to_tsvector` + `plainto_tsquery`) that finds exact keyword matches with a GIN index.

This project runs **both searches in parallel** and merges results with **RRF**:

```
combined_score = Σ( 1 / (k + rank_in_list) )   for each result list
```

Best of both worlds: **semantic recall** + **keyword precision**.

</details>

<details>
<summary><b>📊 Cross-Encoder Reranking</b></summary>

After merging vector + FTS results you have up to 50 candidate chunks. A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads each `(question, chunk)` pair *together* and scores how well they match.

This is far more accurate than cosine similarity because the model sees both texts simultaneously and understands their relationship. The **top-5 highest-scoring chunks** go to the LLM.

The model is cached in memory after first load.

</details>

<details>
<summary><b>📄 Docling — Structural PDF Parsing</b></summary>

Docling goes beyond simple text extraction. It understands document structure:

- **Text blocks** — paragraphs, headings, captions — with page number and bounding box
- **Tables** — identified visually and exported to Markdown, preserving row/column structure
- **Pictures/figures** — extracted as images for vision LLM processing

**OCR fallback:** Scanned PDFs (image-only, no text layer) return zero chunks on first pass. The pipeline detects this and automatically retries with Tesseract OCR enabled.

</details>

<details>
<summary><b>👁️ Vision LLM Enrichment</b></summary>

For figures and diagrams, the pipeline sends the image (as base64) to **Google Gemini 2.0 Flash** via OpenRouter and asks it to describe the image in detail for search indexing.

The description is stored as a `chunk_type="vision"` chunk. When a user asks about a figure, the answer is grounded in the AI-generated description rather than raw pixels.

</details>

<details>
<summary><b>⚙️ Celery + Redis — Background Processing</b></summary>

Parsing and embedding a PDF takes 10–90 seconds. The upload endpoint cannot block the browser that long. Instead:

1. FastAPI saves the file and returns immediately with `status = queued`
2. A Celery task is pushed to the Redis queue
3. A separate Celery worker process picks it up and runs the full ingestion pipeline
4. The UI polls every 4 seconds and updates the status badge when the document is `ready`

> **Windows note:** Celery must run with `--pool=solo` because Windows does not support Unix-style `fork()`.

</details>

<details>
<summary><b>🗄️ MinIO — Object Storage</b></summary>

Raw PDF bytes are stored in MinIO, an S3-compatible object store that runs locally in Docker.

- FastAPI saves the storage key (e.g. `user-id/uuid/filename.pdf`) in Postgres, and the actual bytes in MinIO
- The Celery worker downloads from MinIO when processing
- Uses the standard `boto3` S3 client — **switching to real AWS S3 in production requires only an env var change, no code changes**

</details>

<details>
<summary><b>🔐 JWT Authentication</b></summary>

`fastapi-users` handles registration, login, and JWT auth. After login the server returns an `access_token`. The browser stores it in `localStorage` and sends it as a `Bearer` header with every request. All data queries are additionally filtered by `user.id` — the JWT and the DB filter together enforce per-user isolation.

</details>

<details>
<summary><b>🗃️ SQLAlchemy + Alembic</b></summary>

SQLAlchemy lets you define tables as Python classes and query them in Python rather than raw SQL. Alembic tracks schema changes as versioned migration scripts so the database can be upgraded without losing data. Every schema change gets its own numbered file under `backend/alembic/versions/`.

</details>

---

## 🗂️ Project Structure

```
DocLens/
├── 🐳 docker-compose.yml          # Postgres, Redis, MinIO, Qdrant containers
└── 📦 backend/
    ├── .env.example               # config template — copy to .env and fill in
    ├── pyproject.toml             # Python dependencies (managed by uv)
    ├── alembic.ini                # Alembic config
    ├── alembic/versions/          # database migration scripts
    └── app/
        ├── main.py                # FastAPI app entrypoint
        ├── worker.py              # Celery app + task definition
        ├── users.py               # fastapi-users wiring
        ├── core/config.py         # Pydantic settings (reads .env)
        ├── db/session.py          # async engine + session factory
        ├── models/
        │   ├── user.py            # User SQLAlchemy model
        │   ├── document.py        # Document + DocumentChunk models
        │   └── chat.py            # ChatSession + Message models
        ├── schemas/
        │   ├── document.py        # DocumentOut, CitationOut
        │   ├── chat.py            # ChatRequest, ChatResponse
        │   └── user.py            # UserRead/Create
        ├── api/
        │   ├── documents.py       # /api/documents CRUD + reprocess
        │   ├── chat.py            # /api/sessions + /api/chat
        │   └── auth.py            # fastapi-users auth router
        ├── services/
        │   ├── storage.py         # MinIO upload / download / delete
        │   ├── embeddings.py      # sentence-transformers + cross-encoder
        │   ├── ingestion.py       # Docling → chunk → embed → Qdrant
        │   ├── llm.py             # OpenRouter chat + vision API calls
        │   ├── retrieval.py       # hybrid search + RRF + reranking
        │   └── vector_store.py    # Qdrant upsert / search / delete
        └── static/
            ├── index.html         # single-page app shell
            ├── css/style.css      # dark theme styles
            └── js/app.js          # fetch-based client (no build step)
```

---

## 🚀 Quick Start (Docker — Recommended)

Docker handles Postgres, Redis, MinIO, and Qdrant. You only need Python + `uv`.

### Prerequisites

| Requirement | Link |
|-------------|------|
| 🐳 Docker Desktop | [docker.com](https://www.docker.com/products/docker-desktop/) |
| 🐍 Python 3.11+ | [python.org](https://python.org) |
| ⚡ uv (package manager) | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| 🔑 OpenRouter API key | [openrouter.ai/keys](https://openrouter.ai/keys) — free tier available |

### Step-by-Step

**① Start infrastructure containers**

```bash
cd DocLens
docker compose up -d
```

Starts: **PostgreSQL** `:5432` · **Redis** `:6379` · **MinIO** `:9000/:9001` · **Qdrant** `:6333`

---

**② Install Python dependencies**

```bash
cd backend
uv sync
```

Creates a `.venv` and installs everything from `pyproject.toml`.

---

**③ Configure environment variables**

```bash
cp .env.example .env
# then edit .env with your values
```

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/multimodal_rag` |
| `REDIS_URL` | `redis://localhost:6379/0` |
| `JWT_SECRET` | `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `OPENROUTER_API_KEY` | Your key from openrouter.ai |
| `OPENROUTER_MODEL` | `google/gemini-2.5-flash-preview-05-20:free` |
| `OPENROUTER_VISION_MODEL` | `google/gemini-2.0-flash-exp:free` |
| `MINIO_ENDPOINT_URL` | `http://localhost:9000` |
| `QDRANT_URL` | `http://localhost:6333` |

---

**④ Run database migrations**

```bash
uv run alembic upgrade head
```

---

**⑤ Start the app — 3 terminals**

```bash
# Terminal 1 — FastAPI server
uv run uvicorn app.main:app --reload

# Terminal 2 — Celery worker (PDF processing)
uv run celery -A app.worker worker --loglevel=info --pool=solo

# Terminal 3 — (already running) Docker infrastructure
docker compose up -d
```

> **Windows:** `--pool=solo` is required. Linux/macOS can omit it for better concurrency.

**Open → [http://127.0.0.1:8000](http://127.0.0.1:8000)** 🎉

---

## 🛠️ Manual Setup (Without Docker)

<details>
<summary>Click to expand manual installation instructions</summary>

### PostgreSQL

```bash
# macOS
brew install postgresql@16 && brew services start postgresql@16

# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib && sudo systemctl start postgresql

# Windows — download from https://www.postgresql.org/download/windows/
```

Then create the database:

```sql
CREATE USER raguser WITH PASSWORD 'ragpass';
CREATE DATABASE ragdb OWNER raguser;
```

### Redis

```bash
# macOS
brew install redis && brew services start redis

# Ubuntu/Debian
sudo apt install redis-server && sudo systemctl start redis

# Windows — https://github.com/tporadowski/redis/releases or WSL2
```

### MinIO

```bash
# Linux/macOS
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
  ./minio server ./miniodata --console-address :9001

# Windows PowerShell
$env:MINIO_ROOT_USER="minioadmin"; $env:MINIO_ROOT_PASSWORD="minioadmin"
.\minio.exe server .\miniodata --console-address :9001
```

Log in at [http://localhost:9001](http://localhost:9001) and create a bucket named `documents`.

### Qdrant

Download from [github.com/qdrant/qdrant/releases](https://github.com/qdrant/qdrant/releases) and run:

```bash
./qdrant        # Linux/macOS
.\qdrant.exe    # Windows
```

Runs on port `6333` by default.

### Python setup

```bash
cd backend
uv sync
cp .env.example .env   # fill in your values
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
# second terminal:
uv run celery -A app.worker worker --loglevel=info --pool=solo
```

</details>

---

## 📖 Using the App

| Action | How |
|--------|-----|
| 📝 **Register** | Click Register tab → enter email + password |
| 🔑 **Log in** | JWT token saved in browser; page reloads automatically |
| 📤 **Upload PDF** | Click **+** in the Documents panel. Badge: `Queued → Processing → Ready` (10–90s) |
| 💬 **Ask a question** | Type in the message box. Answer cites file · page · content type |
| 🗑️ **Delete a document** | Click **×** — removes vectors, chunks, and the raw file |
| 🔄 **Retry failed** | Click **↺** on a Failed document to reprocess from scratch |

---

## 🔧 Troubleshooting

<details>
<summary><b>📌 Document stuck on "Queued"</b></summary>

The Celery worker is not running or not connected to Redis.

```bash
redis-cli ping   # should print PONG
```

Check terminal 2 for worker errors.
</details>

<details>
<summary><b>📌 InterfaceError: cannot perform operation: another operation is in progress</b></summary>

This is the asyncpg event-loop binding issue. It only appears if `engine.dispose()` is missing from the Celery task. The `worker.py` in this repo already includes the fix — ensure you are running the latest version.
</details>

<details>
<summary><b>📌 422 Unprocessable Entity on PDF upload</b></summary>

Usually the browser sent the wrong `Content-Type`. Hard refresh (`Ctrl+Shift+R`) to clear cached JavaScript.
</details>

<details>
<summary><b>📌 Vision chunks not appearing</b></summary>

The vision model must support image inputs. Check that `OPENROUTER_VISION_MODEL` is set to a multimodal model (e.g. `google/gemini-2.0-flash-exp:free`). The text model (`OPENROUTER_MODEL`) may not support images.
</details>

<details>
<summary><b>📌 OpenRouter 429 / 504 errors</b></summary>

Free-tier models have rate limits. Wait 30–60 seconds and retry. Switch to another `:free` model at [openrouter.ai/models](https://openrouter.ai/models) if it persists.
</details>

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| ⚡ API server | FastAPI (Python) | Async REST API, static file serving |
| 🔐 Auth | fastapi-users + JWT | Registration, login, per-request identity |
| 🐘 Database | PostgreSQL + asyncpg | Users, documents, chunks, chat history |
| 🔄 Migrations | Alembic | Schema version control |
| ⚙️ Task queue | Celery + Redis | Async PDF ingestion pipeline |
| 🗄️ Object storage | MinIO (S3-compatible) | Raw PDF file storage |
| 🔷 Vector DB | Qdrant | Nearest-neighbour embedding search |
| 📄 PDF parsing | Docling | Text, table, and figure extraction |
| 🧠 Embeddings | BAAI/bge-small-en-v1.5 | Local 384-dim semantic embeddings |
| 📊 Reranking | ms-marco-MiniLM-L-6-v2 | Cross-encoder precision reranking |
| 💬 LLM / chat | OpenRouter (configurable) | Grounded answers from retrieved context |
| 👁️ Vision LLM | Gemini 2.0 Flash (OpenRouter) | Figure/diagram description for indexing |
| 🌐 Frontend | Vanilla HTML + CSS + JS | No build step, served by FastAPI |
| ✨ 3D background | Three.js (CDN) | Animated particle scene |