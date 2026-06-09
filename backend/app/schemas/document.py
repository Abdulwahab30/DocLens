import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    error_message: str | None
    page_count: int | None
    created_at: datetime


class CitationOut(BaseModel):
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_id: uuid.UUID
    chunk_type: str | None = None
