import asyncio
import io
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.user import User
from app.schemas.document import DocumentOut
from app.services import storage, vector_store
from app.users import current_active_user
from app.worker import process_document

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_CONTENT_TYPE = "application/pdf"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB


async def _get_owned(document_id: uuid.UUID, user: User, db: AsyncSession) -> Document:
    """Fetch a document scoped to the requesting user; raises 404 if missing or not owned."""
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    if file.content_type != ALLOWED_CONTENT_TYPE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported right now.")

    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is too large (20MB limit).")

    document = Document(
        user_id=user.id,
        filename=file.filename or "document.pdf",
        content_type=file.content_type,
        size_bytes=len(body),
        storage_key=f"{user.id}/{uuid.uuid4()}/{file.filename or 'document.pdf'}",
        status=DocumentStatus.queued,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    await asyncio.to_thread(storage.upload_fileobj, io.BytesIO(body), document.storage_key, document.content_type)
    process_document.delay(str(document.id))
    return document


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[Document]:
    result = await db.execute(
        select(Document).where(Document.user_id == user.id).order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    return await _get_owned(document_id, user, db)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    doc = await _get_owned(document_id, user, db)
    # Remove from MinIO and Qdrant first; Postgres cascade handles chunk rows.
    await asyncio.to_thread(storage.delete_object, doc.storage_key)
    await asyncio.to_thread(vector_store.delete_document_chunks, document_id)
    await db.delete(doc)
    await db.commit()


@router.patch("/{document_id}/reprocess", response_model=DocumentOut)
async def reprocess_document(
    document_id: uuid.UUID,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    doc = await _get_owned(document_id, user, db)
    # Clear existing vectors and chunks so we get a clean re-ingest.
    await asyncio.to_thread(vector_store.delete_document_chunks, document_id)
    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    doc.status = DocumentStatus.queued
    doc.error_message = None
    await db.commit()
    await db.refresh(doc)
    process_document.delay(str(doc.id))
    return doc
