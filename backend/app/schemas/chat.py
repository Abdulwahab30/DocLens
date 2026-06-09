import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.chat import MessageRole
from app.schemas.document import CitationOut


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: MessageRole
    content: str
    citations: list[CitationOut] | None = None
    created_at: datetime


class ChatRequest(BaseModel):
    session_id: uuid.UUID
    content: str


class ChatResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut
