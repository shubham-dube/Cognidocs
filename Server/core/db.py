# app/core/db.py
from typing import AsyncGenerator
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from core.config import settings

_mongo_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def get_client() -> AsyncIOMotorClient:
    """
    Lazily create a Motor client. Reuse across imports.
    """
    global _mongo_client
    if _mongo_client is None:
        if not settings.MONGO_URI:
            raise RuntimeError("MONGO_URI not set in environment")
        _mongo_client = AsyncIOMotorClient(settings.MONGO_URI, tz_aware=True)
    return _mongo_client


def get_db() -> AsyncIOMotorDatabase:
    """
    Return the configured database instance.
    """
    global _db
    if _db is None:
        client = get_client()
        _db = client[settings.MONGO_DB]
    return _db


async def close_client() -> None:
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
