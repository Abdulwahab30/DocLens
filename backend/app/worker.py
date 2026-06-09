"""Celery application for the async document-ingestion pipeline.

Uses the existing Redis container as both broker and result backend — no new queue
infrastructure, just a worker process consuming from it.

Run with (Windows needs the `solo` pool — the default "prefork" pool requires `fork()`,
which isn't available on Windows):

    uv run celery -A app.worker worker --loglevel=info --pool=solo
"""

import asyncio

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery("multimodal_rag", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_serializer="json", accept_content=["json"], result_serializer="json")


@celery_app.task(name="process_document")
def process_document(document_id: str) -> None:
    """Entry point Celery calls. Tasks are synchronous by design; we bridge into our
    existing async DB/service layer with a single `asyncio.run` per task rather than
    maintaining a second, sync database stack."""
    from app.db.session import engine
    from app.services.ingestion import process_document_async

    async def _run() -> None:
        try:
            await process_document_async(document_id)
        finally:
            # asyncpg connections are bound to the event loop that created them. Each
            # Celery task gets a fresh loop via asyncio.run, so pooled connections from
            # a previous task's (now-closed) loop are unusable in this one — reusing
            # them raises "cannot perform operation: another operation is in progress"
            # or "'NoneType' object has no attribute 'send'". Disposing here forces the
            # next task to open connections bound to its own loop.
            await engine.dispose()

    asyncio.run(_run())
