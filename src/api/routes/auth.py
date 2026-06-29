"""
src/api/routes/auth.py
───────────────────────
Routes d'authentification : register, login.
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from src.utils.auth_utils import hash_password, verify_password, create_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterInput(BaseModel):
    email:      str
    password:   str
    first_name: Optional[str] = None


class LoginInput(BaseModel):
    email:    str
    password: str


class AuthResponse(BaseModel):
    token:      str
    user_id:    int
    first_name: Optional[str] = None


@router.post("/register", response_model=AuthResponse)
async def register(payload: RegisterInput, request: Request):
    """Crée un nouveau compte utilisateur."""
    pool = request.app.state.db_pool
    if not pool:
        raise HTTPException(status_code=503, detail="Base de données non disponible.")

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1", payload.email
        )
        if existing:
            raise HTTPException(status_code=409, detail="Email déjà utilisé.")

        user = await conn.fetchrow(
            """
            INSERT INTO users (email, password_hash, first_name)
            VALUES ($1, $2, $3)
            RETURNING id, first_name
            """,
            payload.email,
            hash_password(payload.password),
            payload.first_name,
        )

    token = create_token(user["id"])
    return AuthResponse(token=token, user_id=user["id"], first_name=user["first_name"])


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginInput, request: Request):
    """Connexion avec email + mot de passe."""
    pool = request.app.state.db_pool
    if not pool:
        raise HTTPException(status_code=503, detail="Base de données non disponible.")

    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, password_hash, first_name FROM users WHERE email = $1",
            payload.email,
        )

    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect.")

    token = create_token(user["id"])
    return AuthResponse(token=token, user_id=user["id"], first_name=user["first_name"])