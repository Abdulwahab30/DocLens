"""Local embedding model wrapper (sentence-transformers).

The model is loaded once per process and reused — loading it is the slow part (it has to
read ~130MB of weights from disk, or download them from HuggingFace on first run and
cache them under the user's home directory). `encode` itself is CPU-bound and synchronous,
which is fine: this is only ever called from the Celery worker (already sync) or wrapped
in `asyncio.to_thread` from the async query path in `app/api/chat.py`.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import get_settings

settings = get_settings()


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings, returning one vector per input in the same order."""
    vectors = get_embedding_model().encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


@lru_cache
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def score_pairs(query: str, texts: list[str]) -> list[float]:
    """Return a relevance score for each (query, text) pair — higher is more relevant."""
    pairs = [[query, t] for t in texts]
    return get_cross_encoder().predict(pairs).tolist()
