from fastapi_users.db import SQLAlchemyBaseUserTableUUID

from app.db.base import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    """Account record. fastapi-users supplies id/email/hashed_password/is_active/is_verified."""

    __tablename__ = "users"
