from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.services import storage, vector_store

app = FastAPI(title="Multimodal Document Intelligence RAG — Phase 2")

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
async def ensure_ingestion_infra() -> None:
    """Create the MinIO bucket and Qdrant collection if they don't exist yet, so a fresh
    stack works without a manual setup step."""
    storage.ensure_bucket()
    vector_store.ensure_collection()


STATIC_DIR = Path(__file__).parent / "static"
# html=True serves index.html for "/" and lets the SPA handle client-side routes.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
