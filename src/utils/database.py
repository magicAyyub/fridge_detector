"""
src/utils/database.py
──────────────────────
Pool de connexion NeonDB via asyncpg.

Installation :
  pip install asyncpg python-dotenv

Variable d'environnement :
  DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require
"""

import os
import asyncpg
import logging
from typing import Optional

logger = logging.getLogger(__name__)
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL manquante dans les variables d'environnement.")
        _pool = await asyncpg.create_pool(dsn=url, min_size=2, max_size=10, ssl="require")
        logger.info("✅ Pool NeonDB créé.")
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Pool NeonDB fermé.")