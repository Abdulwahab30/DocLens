"""Qdrant wrapper for chunk embeddings.

Design note: Qdrant is *not* the source of truth for chunk content. Postgres holds the
chunk text and provenance (page_number, bbox); Qdrant stores only the embedding vector
plus the minimal payload needed to (a) filter by owner and (b) look the chunk back up —
`user_id`, `document_id`, `chunk_id`. After a similarity search we fetch the full chunk
rows from Postgres by `chunk_id`. This keeps the two stores in sync trivially: Qdrant
points are deleted whenever their owning Document is deleted, and never hold data that
could drift from Postgres.

The `user_id` filter on every upsert and search is the per-user isolation boundary for
retrieval — the vector-search analogue of `_get_owned_session` in `app/api/chat.py`.
"""

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings

settings = get_settings()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection() -> None:
    """Create the chunk collection if it doesn't exist. Safe to call repeatedly."""
    client = get_qdrant_client()
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qmodels.VectorParams(
                size=settings.embedding_dimensions,
                distance=qmodels.Distance.COSINE,
            ),
        )


def upsert_chunks(
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    chunk_ids: list[uuid.UUID],
    vectors: list[list[float]],
) -> None:
    """Store one embedding point per chunk, tagged with owner/document/chunk identifiers."""
    points = [
        qmodels.PointStruct(
            id=str(chunk_id),
            vector=vector,
            payload={"user_id": str(user_id), "document_id": str(document_id), "chunk_id": str(chunk_id)},
        )
        for chunk_id, vector in zip(chunk_ids, vectors)
    ]
    get_qdrant_client().upsert(collection_name=settings.qdrant_collection, points=points)


def delete_document_chunks(document_id: uuid.UUID) -> None:
    """Remove all vector points belonging to a document (e.g. on re-ingestion or deletion)."""
    get_qdrant_client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=str(document_id)))]
            )
        ),
    )


def search(user_id: uuid.UUID, query_vector: list[float], top_k: int = 5) -> list[uuid.UUID]:
    """Return the chunk_ids of the top-k chunks owned by `user_id`, most similar first.

    The `user_id` filter is mandatory and non-optional — it is the only thing standing
    between one user's documents and another's in the retrieval path.
    """
    results = get_qdrant_client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=str(user_id)))]
        ),
        limit=top_k,
    )
    return [uuid.UUID(point.payload["chunk_id"]) for point in results.points]
