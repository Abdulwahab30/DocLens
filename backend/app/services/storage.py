"""Thin wrapper around boto3's S3 client, pointed at our local MinIO container.

Using the standard `boto3` S3 client (rather than a MinIO-specific SDK) means this
module would work unchanged against real AWS S3 later — only `MINIO_ENDPOINT_URL` and
the credentials in `Settings` would need to change.

boto3 is synchronous. The upload endpoint (async FastAPI) calls these helpers through
`asyncio.to_thread` so a slow upload doesn't block the event loop; the Celery worker
calls them directly since its tasks are already synchronous.
"""

from functools import lru_cache
from typing import BinaryIO

import boto3
from botocore.client import Config as BotoConfig

from app.core.config import get_settings

settings = get_settings()


@lru_cache
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint_url,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket() -> None:
    """Create the configured bucket if it doesn't already exist. Safe to call repeatedly."""
    client = get_s3_client()
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    if settings.minio_bucket not in existing:
        client.create_bucket(Bucket=settings.minio_bucket)


def upload_fileobj(fileobj: BinaryIO, storage_key: str, content_type: str) -> None:
    get_s3_client().upload_fileobj(
        fileobj,
        settings.minio_bucket,
        storage_key,
        ExtraArgs={"ContentType": content_type},
    )


def download_fileobj(storage_key: str, fileobj: BinaryIO) -> None:
    get_s3_client().download_fileobj(settings.minio_bucket, storage_key, fileobj)


def delete_object(storage_key: str) -> None:
    get_s3_client().delete_object(Bucket=settings.minio_bucket, Key=storage_key)
