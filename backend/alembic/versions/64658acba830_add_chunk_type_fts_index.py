"""add_chunk_type_fts_index

Revision ID: 64658acba830
Revises: 5f1a928a54d5
Create Date: 2026-06-09 10:03:52.526423

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '64658acba830'
down_revision: Union[str, Sequence[str], None] = '5f1a928a54d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('document_chunks', sa.Column('chunk_type', sa.String(length=20), nullable=True))
    op.execute("CREATE INDEX ix_document_chunks_text_fts ON document_chunks USING GIN (to_tsvector('english', text))")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_text_fts")
    op.drop_column('document_chunks', 'chunk_type')
