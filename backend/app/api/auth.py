from fastapi import APIRouter

from app.schemas.user import UserCreate, UserRead, UserUpdate
from app.users import auth_backend, fastapi_users

router = APIRouter(prefix="/api/auth", tags=["auth"])

# POST /api/auth/jwt/login, POST /api/auth/jwt/logout
router.include_router(fastapi_users.get_auth_router(auth_backend), prefix="/jwt")

# POST /api/auth/register
router.include_router(fastapi_users.get_register_router(UserRead, UserCreate))

# GET/PATCH /api/auth/users/me, GET/PATCH/DELETE /api/auth/users/{id}
router.include_router(fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users")
