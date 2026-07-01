"""
src/utils/auth_utils.py
────────────────────────
JWT + hachage des mots de passe + dependency FastAPI.

Installation :
  pip install "passlib[bcrypt]" "python-jose[cryptography]"
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, Header, HTTPException
from jose import JWTError, jwt
import bcrypt

SECRET_KEY        = os.environ.get("JWT_SECRET", "change_me_in_production")
ALGORITHM         = "HS256"
TOKEN_EXPIRE_DAYS = 30


def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        pwd_bytes = plain.encode("utf-8")
        hashed_bytes = hashed.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hashed_bytes)
    except Exception:
        return False


def create_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": str(user_id), "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[int]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None


async def get_current_user_id(authorization: str = Header(...)) -> int:
    """Dependency FastAPI — extrait user_id depuis Authorization: Bearer <token>"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token manquant ou invalide.")
    token = authorization[7:]
    user_id = decode_token(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Token expiré ou invalide.")
    return user_id


async def get_optional_user_id(authorization: Optional[str] = Header(default=None)) -> Optional[int]:
    """Dependency optionnelle — retourne None si pas de token."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return decode_token(authorization[7:])