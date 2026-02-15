import json
import os
from typing import Protocol


class ReplacementStore(Protocol):
    """Protocol for storing PII replacement mappings."""

    async def save(
        self, request_id: str, replacement_map: dict[str, str], ttl: int = 3600
    ) -> None: ...

    async def get(self, request_id: str) -> dict[str, str] | None: ...

    async def delete(self, request_id: str) -> None: ...


class InMemoryReplacementStore:
    """In-memory store for PII replacement mappings with TTL support."""

    def __init__(self):
        self._store: dict[str, tuple[dict[str, str], float]] = {}

    async def save(
        self, request_id: str, replacement_map: dict[str, str], ttl: int = 3600
    ) -> None:
        import time

        expiry = time.time() + ttl
        self._store[request_id] = (replacement_map, expiry)

    async def get(self, request_id: str) -> dict[str, str] | None:
        import time

        if request_id not in self._store:
            return None
        data, expiry = self._store[request_id]
        if time.time() > expiry:
            del self._store[request_id]
            return None
        return data

    async def delete(self, request_id: str) -> None:
        self._store.pop(request_id, None)


class RedisReplacementStore:
    """Redis-backed store for PII replacement mappings."""

    def __init__(self, redis_client):
        self._redis = redis_client
        self._prefix = "pii:replacement:"

    async def save(
        self, request_id: str, replacement_map: dict[str, str], ttl: int = 3600
    ) -> None:
        key = f"{self._prefix}{request_id}"
        await self._redis.setex(key, ttl, json.dumps(replacement_map))

    async def get(self, request_id: str) -> dict[str, str] | None:
        key = f"{self._prefix}{request_id}"
        data = await self._redis.get(key)
        if data is None:
            return None
        return json.loads(data)

    async def delete(self, request_id: str) -> None:
        key = f"{self._prefix}{request_id}"
        await self._redis.delete(key)


async def init_replacement_store() -> ReplacementStore:
    """Initialize replacement store - Redis if available, else in-memory."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(redis_url)
            await client.ping()
            print("Using Redis for replacement storage")
            return RedisReplacementStore(client)
        except ImportError:
            print("redis package not installed, using in-memory storage")
        except Exception as e:
            print(f"Redis unavailable ({e}), falling back to in-memory storage")

    print("Using in-memory replacement storage")
    return InMemoryReplacementStore()
