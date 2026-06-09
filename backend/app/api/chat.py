import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.models.chat import ChatSession, Message, MessageRole
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse, MessageOut, SessionOut
from app.services.llm import get_assistant_reply, get_grounded_reply
from app.services.retrieval import retrieve_relevant_chunks
from app.users import current_active_user

router = APIRouter(prefix="/api", tags=["chat"])


async def _get_owned_session(session_id: uuid.UUID, user: User, db: AsyncSession) -> ChatSession:
    """Fetch a chat session, scoped to the requesting user.

    This per-user filter is the security boundary: a 404 here (rather than leaking
    existence) is what stops one user from ever touching another user's conversations.
    """
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.id == session_id, ChatSession.user_id == user.id)
        .options(selectinload(ChatSession.messages))
    )
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return chat_session


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSession:
    chat_session = ChatSession(user_id=user.id)
    db.add(chat_session)
    await db.commit()
    await db.refresh(chat_session)
    return chat_session


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.created_at.desc())
    )
    return list(result.scalars().all())


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
async def get_session_messages(
    session_id: uuid.UUID,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[Message]:
    chat_session = await _get_owned_session(session_id, user, db)
    return list(chat_session.messages)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    chat_session = await _get_owned_session(payload.session_id, user, db)

    # Append (not just db.add) so the loaded `chat_session.messages` collection reflects the
    # new message in-memory too — we need that to build the LLM history below.
    user_message = Message(role=MessageRole.user, content=payload.content)
    chat_session.messages.append(user_message)
    if len(chat_session.messages) == 1:
        chat_session.title = payload.content[:60].strip()
    await db.commit()
    await db.refresh(user_message)

    # Replay full history (including the message we just added) so the model has context.
    history = [{"role": m.role.value, "content": m.content} for m in chat_session.messages]

    # Always attempt retrieval first; fall back to an ungrounded reply when the user has
    # no ingested documents or nothing in them is relevant to this question.
    retrieved_chunks = await retrieve_relevant_chunks(payload.content, user.id, db)
    if retrieved_chunks:
        reply_text = await get_grounded_reply(history, retrieved_chunks)
        citations = [
            {
                "document_id": str(chunk["document_id"]),
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "chunk_id": str(chunk["chunk_id"]),
                "chunk_type": chunk.get("chunk_type", "text"),
            }
            for chunk in retrieved_chunks
        ]
    else:
        reply_text = await get_assistant_reply(history)
        citations = None

    assistant_message = Message(role=MessageRole.assistant, content=reply_text, citations=citations)
    chat_session.messages.append(assistant_message)
    await db.commit()
    await db.refresh(assistant_message)

    return ChatResponse(user_message=user_message, assistant_message=assistant_message)
