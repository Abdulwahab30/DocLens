"""Retrieval step of the RAG query pipeline: turn a user's question into the document
chunks (with provenance) that ground the assistant's answer.

Phase 3 upgrade — two-stage pipeline:
  1. Candidate generation: dense vector search (Qdrant, top-30) + Postgres full-text
     search (top-20), merged with Reciprocal Rank Fusion (RRF). This gives keyword
     recall (FTS) on top of semantic recall (dense vectors) without touching Qdrant's
     schema or re-indexing existing data.
  2. Reranking: a cross-encoder (ms-marco-MiniLM-L-6-v2) scores every (query, chunk)
     pair from the merged candidate set and keeps the top-5. Cross-encoders read both
     sides jointly so they catch relevance that cosine similarity misses.

Postgres is still the source of truth for chunk text and provenance; Qdrant stores only
vectors + a `user_id` payload for the isolation boundary filter.
"""

import asyncio
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.document import Document, DocumentChunk
from app.services import vector_store
from app.services.embeddings import embed_query, score_pairs

TOP_K_VECTOR = 30
TOP_K_FTS = 20
TOP_K_FINAL = 5


def _reciprocal_rank_fusion(*ranked_lists: list[uuid.UUID], k: int = 60) -> list[uuid.UUID]:
    """Merge ranked lists with RRF, returning a deduplicated list ordered by fused score."""
    scores: dict[uuid.UUID, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


async def retrieve_relevant_chunks(query: str, user_id: uuid.UUID, db: AsyncSession) -> list[dict]:
    """Return up to TOP_K_FINAL chunks owned by `user_id` that are most relevant to `query`.

    Each result dict has: chunk_id, document_id, filename, page_number, text, chunk_type.
    Returns an empty list when the user has no ready documents or nothing matches.
    """
    query_vector = await asyncio.to_thread(embed_query, query)

    # 1. Dense vector search — returns chunk IDs in cosine-similarity order.
    vector_ids = await asyncio.to_thread(vector_store.search, user_id, query_vector, TOP_K_VECTOR)

    # 2. Postgres full-text search — keyword recall for terms with low semantic overlap.
    fts_result = await db.execute(
        select(DocumentChunk)
        .join(Document)
        .where(Document.user_id == user_id)
        .where(
            func.to_tsvector("english", DocumentChunk.text).op("@@")(
                func.plainto_tsquery("english", query)
            )
        )
        .order_by(
            func.ts_rank(
                func.to_tsvector("english", DocumentChunk.text),
                func.plainto_tsquery("english", query),
            ).desc()
        )
        .limit(TOP_K_FTS)
        .options(selectinload(DocumentChunk.document))
    )
    fts_chunks = fts_result.scalars().all()
    fts_ids = [c.id for c in fts_chunks]

    # 3. RRF merge — combines the two ranked lists into one deduplicated ordering.
    merged_ids = _reciprocal_rank_fusion(vector_ids, fts_ids)
    if not merged_ids:
        return []

    # 4. Fetch full rows for the merged candidate set (FTS rows already loaded; fetch the rest).
    fts_by_id = {c.id: c for c in fts_chunks}
    missing_ids = [i for i in merged_ids if i not in fts_by_id]
    if missing_ids:
        extra = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.id.in_(missing_ids))
            .options(selectinload(DocumentChunk.document))
        )
        extra_by_id = {c.id: c for c in extra.scalars().all()}
    else:
        extra_by_id = {}

    all_by_id = {**fts_by_id, **extra_by_id}
    candidates = [all_by_id[i] for i in merged_ids if i in all_by_id]
    if not candidates:
        return []

    # 5. Cross-encoder rerank — scores (query, chunk_text) jointly; far more precise than
    #    cosine similarity. CPU-bound, so it runs in a thread.
    texts = [c.text for c in candidates]
    scores = await asyncio.to_thread(score_pairs, query, texts)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top = [chunk for _, chunk in ranked[:TOP_K_FINAL]]

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "filename": chunk.document.filename,
            "page_number": chunk.page_number,
            "text": chunk.text,
            "chunk_type": chunk.chunk_type or "text",
        }
        for chunk in top
    ]
