"""
Redis key-value storage adapter for RAG query system.

Handles:
- Text chunk content retrieval
- Entity source tracking
- Chunk-entity mapping
- Document status

Adapted from: lightrag/kg/redis_impl.py
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .base import BaseKVStorage

try:
    import redis.asyncio as aioredis
except ImportError:
    raise ImportError("redis not installed. Run: pip install redis>=4.5.0") from None

logger = logging.getLogger(__name__)


@dataclass
class RedisStorage(BaseKVStorage):
    """
    Redis key-value storage implementation for query operations.

    Namespaces:
    - {workspace}_text_chunks: Chunk content (JSON)
    - {workspace}_entity_sources: Entity → Chunks mapping (string)
    - {workspace}_chunk_entities: Chunk → Entities mapping (set)
    - {workspace}_doc_status: Document metadata (JSON)
    """

    _client: aioredis.Redis | None = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._initialized:
            return

        host = self.config.get("host", "localhost")
        port = self.config.get("port", 6379)
        db = self.config.get("db", 0)
        timeout = self.config.get("timeout", 10)
        max_connections = self.config.get("max_connections", 50)
        password = self.config.get("password") or None

        try:
            # Create Redis connection pool
            pool = aioredis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,  # Auto-decode bytes to strings
                socket_connect_timeout=timeout,
                max_connections=max_connections,
            )

            # Initialize Redis client
            self._client = aioredis.Redis(connection_pool=pool)

            # Test connection
            await self._client.ping()

            # Build full namespace
            self.full_namespace = f"{self.workspace}_{self.namespace}"

            logger.info(f"[{self.workspace}] Connected to Redis: {host}:{port}/{db}, namespace: {self.full_namespace}")
            self._initialized = True

        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to initialize Redis: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            try:
                await self._client.close()
                logger.info(f"[{self.workspace}] Closed Redis connection")
            except Exception as e:
                logger.warning(f"[{self.workspace}] Error closing Redis: {e}")
            finally:
                self._client = None
                self._initialized = False

    def _build_key(self, id: str) -> str:
        """Build full Redis key from ID."""
        return f"{self.full_namespace}:{id}"

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """
        Get value by ID.

        For entity_sources namespace: returns SET members as list
        For other namespaces: returns JSON or string

        Args:
            id: The unique identifier

        Returns:
            Dict with data, or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            key = self._build_key(id)

            # Special handling for entity_sources namespace (SET storage)
            if self.namespace == "entity_sources":
                members = await self._client.smembers(key)
                if not members:
                    return None
                return {"id": id, "value": list(members)}

            # Standard handling for other namespaces (JSON/string storage)
            value = await self._client.get(key)
            if value is None:
                return None

            # Parse JSON if possible
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Not JSON, return as string wrapped in dict
                return {"value": value}

        except Exception as e:
            logger.error(f"[{self.workspace}] get_by_id failed for {id}: {e}")
            raise

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Get values by multiple IDs (batch operation).

        Args:
            ids: List of unique identifiers

        Returns:
            List of dicts with data (order preserved, None values skipped)
        """
        if not self._initialized:
            await self.initialize()

        if not ids:
            return []

        try:
            keys = [self._build_key(id) for id in ids]

            # Special handling for entity_sources namespace (SET storage)
            if self.namespace == "entity_sources":
                async with self._client.pipeline() as pipe:
                    for key in keys:
                        pipe.smembers(key)
                    results = await pipe.execute()

                parsed_results = []
                for i, members in enumerate(results):
                    if members:
                        parsed_results.append({"id": ids[i], "value": list(members)})

                logger.debug(f"[{self.workspace}] get_by_ids returned {len(parsed_results)}/{len(ids)} items")
                return parsed_results

            # Standard handling for other namespaces (JSON/string storage)
            async with self._client.pipeline() as pipe:
                for key in keys:
                    pipe.get(key)
                results = await pipe.execute()

            # Parse results and add IDs
            parsed_results = []
            for i, result in enumerate(results):
                if result is None:
                    continue

                try:
                    data = json.loads(result)
                    # Ensure ID is in the data
                    if "id" not in data and "chunk_id" not in data:
                        data["id"] = ids[i]  # Add the ID from the request
                    parsed_results.append(data)
                except json.JSONDecodeError:
                    # Not JSON, return as string wrapped in dict with ID
                    parsed_results.append({"id": ids[i], "value": result})

            logger.debug(f"[{self.workspace}] get_by_ids returned {len(parsed_results)}/{len(ids)} items")

            return parsed_results

        except Exception as e:
            logger.error(f"[{self.workspace}] get_by_ids failed: {e}")
            raise

    async def get_set_members(self, key: str) -> set[str]:
        """
        Get members of a Redis set.

        Used for chunk_entities namespace where values are sets.

        Args:
            key: The set key (without namespace prefix)

        Returns:
            Set of member strings
        """
        if not self._initialized:
            await self.initialize()

        try:
            full_key = self._build_key(key)
            members = await self._client.smembers(full_key)

            # Convert to Python set (Redis returns set already)
            return members if members else set()

        except Exception as e:
            logger.error(f"[{self.workspace}] get_set_members failed for {key}: {e}")
            raise

    async def scan_keys(self, pattern: str, count: int = 100) -> list[str]:
        """
        Scan for keys matching pattern.

        Args:
            pattern: Pattern to match (e.g., "chunk-*")
            count: Maximum number of keys to return (not just scan hint)

        Returns:
            List of matching keys (without namespace prefix), limited to count
        """
        if not self._initialized:
            await self.initialize()

        try:
            full_pattern = self._build_key(pattern)
            keys = []

            cursor = 0
            while True:
                # Use count as batch size hint
                cursor, batch = await self._client.scan(cursor=cursor, match=full_pattern, count=min(count, 1000))
                keys.extend(batch)

                # Stop if we have enough keys or scan is complete
                if len(keys) >= count or cursor == 0:
                    break

            # Limit to requested count
            keys = keys[:count]

            # Remove namespace prefix
            prefix = f"{self.full_namespace}:"
            stripped_keys = [key.replace(prefix, "", 1) for key in keys]

            logger.debug(f"[{self.workspace}] scan_keys({pattern}) found {len(stripped_keys)} keys")

            return stripped_keys

        except Exception as e:
            logger.error(f"[{self.workspace}] scan_keys failed for {pattern}: {e}")
            raise

    async def get(self, key: str) -> str | None:
        """
        Get raw string value from cache (simple Redis GET).

        Args:
            key: Cache key (will be namespaced)

        Returns:
            String value or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            full_key = self._build_key(key)
            value = await self._client.get(full_key)
            return value
        except Exception as e:
            logger.error(f"[{self.workspace}] get failed for {key}: {e}")
            raise

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        """
        Set raw string value in cache (simple Redis SET).

        Args:
            key: Cache key (will be namespaced)
            value: String value to store
            ex: Expiration time in seconds (optional)
        """
        if not self._initialized:
            await self.initialize()

        try:
            full_key = self._build_key(key)
            if ex:
                await self._client.setex(full_key, ex, value)
            else:
                await self._client.set(full_key, value)
        except Exception as e:
            logger.error(f"[{self.workspace}] set failed for {key}: {e}")
            raise
