# Doc Lens

> Upload PDFs. Ask questions. Get answers grounded in your documents — with page citations,
> table awareness, and figure descriptions powered by a vision AI.

---

## What this project does

You upload one or more PDF files. Behind the scenes the system reads every page, extracts
text paragraphs, converts tables to searchable Markdown, and uses a vision AI to describe
figures and diagrams. When you ask a question, the system finds the most relevant passages
(from any of those three content types), hands them to an LLM, and returns an answer that
cites exactly which page and which document it came from.

This is called **Retrieval-Augmented Generation (RAG)** — the AI does not guess from its
training data; it reads your documents first.

---

## Full Pipeline

```
                        ╔══════════════════════════════════════════════════════╗
                        ║                  BROWSER (UI)                        ║
                        ║  Login / Register  ──►  Chat  ──►  Upload PDF        ║
                        ╚══════════════════════╤═══════════════════╤═══════════╝
                                               │                   │
                               QUESTION        │                   │  PDF file
                                               ▼                   ▼
                        ╔══════════════════════════════════════════════════════╗
                        ║                 FASTAPI  (REST API)                  ║
                        ║  JWT auth ── route guards ── async request handlers  ║
                        ╚═══╤══════════════════════════╤═════════════════════╤═╝
                            │ POST /api/chat            │ POST /api/documents │
                            │                           │                     │
                            │                    ┌──────▼──────┐              │
                            │                    │   POSTGRES   │ ◄────────── ├──► MinIO
                            │                    │  (users,     │             │  (raw PDF
                            │                    │  documents,  │             │   stored)
                            │                    │  chunks,     │             │
                            │                    │  chat logs)  │             │
                            │                    └──────┬──────┘              │
                            │                           │ status = queued     │
                            │                           │ task fired          │
                            │                    ┌──────▼──────┐              │
                            │                    │    REDIS     │              │
                            │                    │ (task queue) │              │
                            │                    └──────┬──────┘              │
                            │                           │                     │
                            │                    ┌──────▼──────────────────────────────────────┐
                            │                    │              CELERY WORKER                   │
                            │                    │                                              │
                            │                    │  1. Download PDF from MinIO                  │
                            │                    │  2. DOCLING parses PDF                       │
                            │                    │       ├─ text items  → "text" chunks         │
                            │                    │       ├─ tables      → Markdown "table" chunks│
                            │                    │       └─ figures     → Vision LLM describes  │
                            │                    │                         → "vision" chunks     │
                            │                    │  3. OCR fallback (if zero text extracted)    │
                            │                    │  4. Save all chunks to Postgres              │
                            │                    │  5. Embed each chunk (sentence-transformers) │
                            │                    │  6. Upsert vectors to Qdrant                 │
                            │                    │  7. Set document status = ready              │
                            │                    └──────────────────────────────────────────────┘
                            │
                            │  [Later — when user asks a question]
                            │
                            ▼
          ╔══════════════════════════════════════════════════════════════╗
          ║                    RETRIEVAL PIPELINE                        ║
          ║                                                              ║
          ║  User question                                               ║
          ║       │                                                      ║
          ║       ├─► Embed question (sentence-transformers)             ║
          ║       │         │                                            ║
          ║       │         ▼                                            ║
          ║       │   Qdrant vector search  ──► top-30 chunk IDs        ║
          ║       │                                                      ║
          ║       └─► Postgres full-text search (FTS)  ──► top-20 IDs  ║
          ║                                                              ║
          ║   Reciprocal Rank Fusion (RRF) merges both lists            ║
          ║            │                                                 ║
          ║            ▼                                                 ║
          ║   Cross-encoder reranker  ──► top-5 best chunks             ║
          ║            │                                                 ║
          ║            ▼                                                 ║
          ║   OpenRouter LLM  (grounded prompt with chunk texts)        ║
          ║            │                                                 ║
          ║            ▼                                                 ║
          ║   Answer + citations  ──►  browser                          ║
          ╚══════════════════════════════════════════════════════════════╝
```

---

## Concepts explained (beginner-friendly)

### What is RAG (Retrieval-Augmented Generation)?

A normal AI chatbot answers from its training data — whatever it learned before it was
released. RAG adds a retrieval step: before answering, the system searches your documents
for the most relevant passages and hands them to the AI as extra context. The AI then
answers based on what your documents actually say, and cites the exact page it used. This
prevents hallucination and keeps answers grounded in your data.

### Embeddings and Vector Search

An "embedding" is a list of numbers that represents the meaning of a sentence. Sentences
with similar meaning get similar numbers, even if they use different words. For example,
"heart attack" and "myocardial infarction" are very different words but end up with nearly
identical embedding vectors.

This project uses `BAAI/bge-small-en-v1.5` from the sentence-transformers library — a
small (130 MB), fast model that runs locally with no API key required. It converts every
text chunk into a 384-dimensional vector.

When you ask a question, the question is also embedded into a vector. The system then
finds the document chunks whose vectors are closest to the question vector — those are the
most semantically relevant passages.

### Qdrant (Vector Database)

Qdrant is a database designed specifically for storing and searching embedding vectors.
Think of it like a regular database index but optimised for "find the nearest neighbours
in high-dimensional space". Each entry in Qdrant stores the vector plus a small payload
(`user_id`, `document_id`, `chunk_id`). The full text of each chunk stays in Postgres.

Every search is filtered by `user_id` so users cannot see each other's documents.

### Full-Text Search (FTS) and Hybrid Retrieval

Semantic vector search is great for concepts but can miss exact keywords. If your
document mentions a product code like "ALPHA-7734", a question containing "ALPHA-7734"
might not be close enough in vector space. Postgres has a built-in full-text search
engine (`to_tsvector` + `plainto_tsquery`) that finds exact keyword matches, with a GIN
index to make it fast.

This project runs both searches in parallel and merges their result lists with
**Reciprocal Rank Fusion (RRF)**, a formula that combines rankings from multiple sources
without needing to put scores on the same scale:

```
combined_score = sum(  1 / (k + rank_in_list)  )  for each result list
```

This gives you the best of both worlds: semantic recall + keyword precision.

### Cross-Encoder Reranking

After merging the vector and FTS results you have up to 50 candidate chunks.
A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads each `(question, chunk)`
pair together and scores how well they match. This is much more accurate than cosine
similarity because the model can look at both texts at once and understand their
relationship. The top-5 highest-scoring chunks go to the LLM.

Loading this model takes a few seconds on first use; after that it's cached in memory.

### Docling (PDF Parsing)

Docling is a PDF parsing library that goes beyond simple text extraction. It understands
the structure of a document:

- **Text blocks** — paragraphs, headings, captions — extracted with page number and
  bounding box coordinates.
- **Tables** — identified visually and converted to Markdown format, which preserves the
  row/column structure that a plain text extractor would lose.
- **Pictures/figures** — extracted as images for further processing.

### OCR Fallback

Some PDFs are scanned images — they contain no actual text, just a photo of each page.
Standard text extraction returns nothing for these. The pipeline automatically detects
this (zero chunks extracted) and retries with OCR (Optical Character Recognition) enabled,
which uses `tesseract` under the hood to read text from the images.

### Vision LLM Enrichment

For figures and diagrams extracted by Docling, the pipeline sends the image (as a base64
string) to a multimodal LLM (Google Gemini 2.0 Flash via OpenRouter) and asks it to
describe the image in detail for search indexing. The description is stored as a
`chunk_type="vision"` chunk. When a user asks about a figure, the answer is grounded in
the AI-generated description rather than the raw pixels.

### Celery + Redis (Background Task Queue)

Parsing and embedding a PDF can take 10–60 seconds. The upload endpoint cannot block the
browser that long. Instead:

1. The FastAPI endpoint saves the file and returns immediately with `status = queued`.
2. It fires a Celery task that goes into a Redis queue.
3. A separate Celery worker process picks up the task and runs the full ingestion pipeline.
4. The UI polls every 4 seconds and updates the status badge when the document is `ready`.

On Windows, Celery must run with `--pool=solo` because Windows does not support Unix-style
`fork()`. This means one task runs at a time per worker process.

### asyncpg Event-Loop Binding

FastAPI uses `asyncio` for concurrent request handling. SQLAlchemy's async driver
(`asyncpg`) binds database connections to the event loop that created them. Celery tasks
each create their own event loop via `asyncio.run()`. If the database connection pool is
reused across different Celery tasks (each with a different loop), connections from the
previous loop become unusable. The fix: `await engine.dispose()` at the end of every task
forces the pool to release all connections so the next task starts fresh.

### FastAPI-Users (JWT Authentication)

`fastapi-users` handles user registration, login, and JWT (JSON Web Token) auth. After
login, the server returns an `access_token`. The browser stores this in `localStorage`
and sends it as a `Bearer` header with every API request. The server verifies the token
and rejects requests with an invalid or missing token. All data queries are additionally
filtered by `user.id` so the token and the DB filter together enforce per-user isolation.

### MinIO (Object Storage)

Raw PDF files are stored in MinIO, an S3-compatible object store that runs locally in
Docker. The FastAPI server stores an upload key (like `user-id/uuid/filename.pdf`) in
Postgres and the actual bytes in MinIO. The Celery worker downloads the file from MinIO
when it processes the task. This keeps large binary files out of the database.

### SQLAlchemy + Alembic (Database ORM and Migrations)

SQLAlchemy is a Python library that lets you define database tables as Python classes and
query them with Python rather than raw SQL. Alembic tracks schema changes (adding columns,
creating indexes) as versioned migration scripts so the database can be upgraded without
losing data. Every schema change gets its own numbered migration file under
`backend/alembic/versions/`.

### Pydantic + FastAPI Schemas

FastAPI uses Pydantic models to validate incoming request bodies and serialise outgoing
responses. If a required field is missing or has the wrong type, FastAPI rejects the
request automatically with a clear error message — no manual validation code needed.

---

## Project structure

```
Multi Modal/
├── docker-compose.yml          # Postgres, Redis, MinIO, Qdrant containers
└── backend/
    ├── .env                    # secrets (not committed)
    ├── .env.example            # template — copy to .env and fill in
    ├── pyproject.toml          # Python dependencies (managed by uv)
    ├── alembic.ini             # Alembic config
    ├── alembic/versions/       # database migration scripts
    └── app/
        ├── main.py             # FastAPI app entrypoint, mounts routes
        ├── worker.py           # Celery app + task definition
        ├── users.py            # fastapi-users wiring
        ├── core/config.py      # Pydantic settings (reads .env)
        ├── db/session.py       # async engine + session factory
        ├── models/
        │   ├── user.py         # User SQLAlchemy model
        │   ├── document.py     # Document + DocumentChunk models
        │   └── chat.py         # ChatSession + Message models
        ├── schemas/
        │   ├── document.py     # DocumentOut, CitationOut Pydantic schemas
        │   ├── chat.py         # ChatRequest, ChatResponse schemas
        │   └── user.py         # UserRead/Create schemas
        ├── api/
        │   ├── documents.py    # /api/documents CRUD + reprocess endpoints
        │   ├── chat.py         # /api/sessions + /api/chat endpoints
        │   └── auth.py         # mounts fastapi-users auth router
        ├── services/
        │   ├── storage.py      # MinIO upload / download / delete wrappers
        │   ├── embeddings.py   # sentence-transformers embed + cross-encoder
        │   ├── ingestion.py    # Docling parse → chunk → embed → Qdrant
        │   ├── llm.py          # OpenRouter chat + vision API calls
        │   ├── retrieval.py    # hybrid search + RRF + reranking
        │   └── vector_store.py # Qdrant upsert / search / delete wrappers
        └── static/
            ├── index.html      # single-page app shell
            ├── css/style.css   # dark theme styles
            └── js/app.js       # fetch-based client (no build step)
```

---

## Running with Docker (recommended)

Docker handles Postgres, Redis, MinIO, and Qdrant for you. You only need to install
Python (and `uv`) for the FastAPI server and Celery worker.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes
  Docker Compose)
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package
  manager
- An [OpenRouter](https://openrouter.ai/keys) API key (free-tier models are available)

### Step-by-step

**1. Start the infrastructure containers**

```bash
cd "Multi Modal"
docker compose up -d
```

This starts:
- PostgreSQL on port 5432
- Redis on port 6379
- MinIO on ports 9000 (API) and 9001 (web console)
- Qdrant on port 6333

**2. Install Python dependencies**

```bash
cd backend
uv sync
```

This creates a `.venv` virtual environment and installs all packages listed in
`pyproject.toml`.

**3. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and set these values:

| Variable | What to put |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb` (matches docker-compose) |
| `REDIS_URL` | `redis://localhost:6379/0` (matches docker-compose) |
| `JWT_SECRET` | Run `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `OPENROUTER_API_KEY` | Your key from https://openrouter.ai/keys |
| `OPENROUTER_MODEL` | `google/gemini-2.5-flash-preview-05-20:free` or another model |
| `OPENROUTER_VISION_MODEL` | `google/gemini-2.0-flash-exp:free` (needs multimodal support) |
| `MINIO_ENDPOINT` | `localhost:9000` |
| `MINIO_ACCESS_KEY` | `minioadmin` |
| `MINIO_SECRET_KEY` | `minioadmin` |
| `MINIO_BUCKET` | `documents` |
| `QDRANT_URL` | `http://localhost:6333` |

**4. Run database migrations**

```bash
uv run alembic upgrade head
```

This creates all tables in Postgres and adds the full-text search index.

**5. Start the FastAPI server** (terminal 1)

```bash
uv run uvicorn app.main:app --reload
```

**6. Start the Celery worker** (terminal 2)

```bash
uv run celery -A app.worker worker --loglevel=info --pool=solo
```

The `--pool=solo` flag is required on Windows. On Linux/macOS you can use the default
pool (remove `--pool=solo`) for better concurrency.

**7. Open the app**

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000), register an account, and start
uploading PDFs.

---

## Running without Docker (manual setup)

If you cannot use Docker, you need to install and run four services yourself.

### Install the services

#### PostgreSQL

- Windows: download the installer from https://www.postgresql.org/download/windows/
- macOS: `brew install postgresql@16 && brew services start postgresql@16`
- Ubuntu/Debian: `sudo apt install postgresql postgresql-contrib && sudo systemctl start postgresql`

After installing, create a database and user:

```sql
CREATE USER raguser WITH PASSWORD 'ragpass';
CREATE DATABASE ragdb OWNER raguser;
```

#### Redis

- Windows: download from https://github.com/tporadowski/redis/releases (unofficial but
  works) or use WSL2 with `sudo apt install redis-server`
- macOS: `brew install redis && brew services start redis`
- Ubuntu/Debian: `sudo apt install redis-server && sudo systemctl start redis`

#### MinIO

Download the binary from https://min.io/download:

```bash
# Linux/macOS
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin ./minio server ./miniodata --console-address :9001

# Windows — download minio.exe, then:
$env:MINIO_ROOT_USER="minioadmin"; $env:MINIO_ROOT_PASSWORD="minioadmin"
.\minio.exe server .\miniodata --console-address :9001
```

MinIO creates a `miniodata` directory for storage. Log in at http://localhost:9001
(user: `minioadmin`, password: `minioadmin`) and create a bucket called `documents`.

#### Qdrant

Download from https://github.com/qdrant/qdrant/releases:

```bash
# Linux/macOS
./qdrant

# Windows
.\qdrant.exe
```

Qdrant runs on port 6333 by default.

### Python setup (same as Docker path)

```bash
cd backend
uv sync                         # or: pip install -r requirements.txt
cp .env.example .env            # then edit .env with your values
uv run alembic upgrade head     # or: python -m alembic upgrade head
```

If you installed with `pip` instead of `uv`, replace `uv run` with `python -m` or
activate the virtual environment first:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
# in a second terminal:
celery -A app.worker worker --loglevel=info --pool=solo
```

---

## Environment variables reference

All variables live in `backend/.env`. The full list with descriptions:

```bash
# PostgreSQL — change host/port if not running locally on default port
DATABASE_URL=postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb

# Redis — used by Celery as the broker and result backend
REDIS_URL=redis://localhost:6379/0

# Secret key for signing JWT tokens — generate once and keep it secret
JWT_SECRET=<run: python -c "import secrets; print(secrets.token_urlsafe(32))">

# OpenRouter (https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# Text/chat model (any model on OpenRouter works)
OPENROUTER_MODEL=google/gemini-2.5-flash-preview-05-20:free
# Vision model — must support image inputs
OPENROUTER_VISION_MODEL=google/gemini-2.0-flash-exp:free

# MinIO (S3-compatible object storage for raw PDFs)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=documents
MINIO_USE_SSL=false

# Qdrant (vector database)
QDRANT_URL=http://localhost:6333
```

---

## Using the app

1. **Register** — click the Register tab, enter an email and password.
2. **Log in** — your JWT token is saved in the browser; the page reloads automatically.
3. **Upload a PDF** — click the "+" button in the Documents panel (left sidebar). The
   status badge starts at "Queued", changes to "Processing…", and finally "Ready".
   Processing takes 10–90 seconds depending on the PDF size and whether vision
   enrichment runs.
4. **Ask a question** — type in the message box. The system searches your documents and
   answers with citations showing the source file, page number, and content type
   (Text / Table / Figure).
5. **Delete a document** — click the × button next to any document. All associated
   vectors, chunks, and the raw file are removed.
6. **Retry a failed document** — click the ↺ button on a Failed document to reprocess it
   from scratch.

---

## Troubleshooting

**Document stuck on "Queued"**
The Celery worker is not running, or is not connected to Redis. Check terminal 2 for
errors. Make sure Redis is running (`redis-cli ping` should print `PONG`).

**`InterfaceError: cannot perform operation: another operation is in progress`**
This is the asyncpg event-loop binding bug. It only appears if `engine.dispose()` is
missing from the Celery task. The code in `worker.py` already includes the fix — if you
see this, ensure you are running the latest version of `worker.py`.

**422 Unprocessable Entity on PDF upload**
Usually means the browser sent the wrong `Content-Type`. A hard refresh (`Ctrl+Shift+R`)
clears the cached old JavaScript. The `app.js` `api()` wrapper is careful to leave
`Content-Type` unset for `FormData` so the browser can add the multipart boundary itself.

**Vision chunks not appearing**
The vision model must support image inputs. Check that `OPENROUTER_VISION_MODEL` is set
to a multimodal model (e.g., `google/gemini-2.0-flash-exp:free`). The text model
(`OPENROUTER_MODEL`) may not support images.

**OpenRouter 429 / 504**
Free-tier models have rate limits. Wait 30–60 seconds and retry. If a model is
consistently unavailable, switch to another `:free` model from
[openrouter.ai/models](https://openrouter.ai/models).

---

## Tech stack summary

| Layer | Technology | Purpose |
|---|---|---|
| API server | FastAPI (Python) | Async REST API, static file serving |
| Auth | fastapi-users + JWT | User registration, login, per-request identity |
| Database | PostgreSQL + asyncpg | Users, documents, chunks, chat history |
| Migrations | Alembic | Schema version control |
| Task queue | Celery + Redis | Async PDF ingestion pipeline |
| Object storage | MinIO (S3-compatible) | Raw PDF file storage |
| Vector database | Qdrant | Nearest-neighbour embedding search |
| PDF parsing | Docling | Text, table, figure extraction |
| Embeddings | sentence-transformers (BAAI/bge-small-en-v1.5) | Local 384-dim text embeddings |
| Reranking | sentence-transformers (ms-marco-MiniLM-L-6-v2) | Cross-encoder result reranking |
| LLM / chat | OpenRouter (configurable model) | Grounded answers from retrieved context |
| Vision LLM | OpenRouter (multimodal model) | Figure/diagram description for indexing |
| Frontend | Vanilla HTML + CSS + JS | No build step, served directly by FastAPI |
| 3D background | Three.js (CDN) | Decorative animated particle scene |






Open three separate terminals in VS Code (Ctrl+ ` → click + to add more) and run one command in each:

Terminal 1 — FastAPI server

cd "e:\Abdul_Wahab\resources\Multi Modal\backend"
uv run uvicorn app.main:app --reload



Terminal 2 — Celery worker (handles PDF processing)

cd "e:\Abdul_Wahab\resources\Multi Modal\backend"
uv run celery -A app.worker worker --loglevel=info --pool=solo



Terminal 3 — Docker (Postgres, Redis, MinIO, Qdrant — only needs to run once)

cd "e:\Abdul_Wahab\resources\Multi Modal"
docker compose up -d