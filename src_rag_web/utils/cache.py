"""
Cache utilities for RAG query system.

Based on LightRAG utils.py cache functions.

Provides caching for:
- LLM responses
- Query results
- Entity extraction results
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheData:
    """Cache data structure.

    Based on LightRAG CacheData (lightrag/utils.py:1067-1075)

    Attributes:
        args_hash: Hash of the arguments/inputs
        content: The cached content (LLM response, query result, etc.)
        prompt: The original prompt or query
        mode: Cache mode (default, query, etc.)
        cache_type: Type of cache (llm, query, entity_extract, etc.)
        chunk_id: Optional chunk ID for chunk-level caching
        queryparam: Optional query parameters dictionary
    """

    args_hash: str
    content: str
    prompt: str
    mode: str = "default"
    cache_type: str = "query"
    chunk_id: str | None = None
    queryparam: dict[str, Any] | None = None


def generate_cache_key(mode: str, cache_type: str, args_hash: str) -> str:
    """Generate a flattened cache key.

    Based on LightRAG cache key generation.

    Args:
        mode: Cache mode (default, query, etc.)
        cache_type: Type of cache (llm, query, entity_extract, etc.)
        args_hash: Hash of arguments

    Returns:
        Flattened cache key string in format: {mode}:{cache_type}:{hash}

    Example:
        >>> generate_cache_key("query", "llm", "abc123")
        "query:llm:abc123"
    """
    return f"{mode}:{cache_type}:{args_hash}"


async def handle_cache(
    hashing_kv,
    args_hash: str,
    prompt: str,
    mode: str = "default",
    cache_type: str = "unknown",
) -> tuple[str, int] | None:
    """Retrieve content from cache if available.

    Based on LightRAG handle_cache (lightrag/utils.py:1032-1064)

    Args:
        hashing_kv: Key-value storage for caching
        args_hash: Hash of the arguments
        prompt: The prompt text (for logging/debugging)
        mode: Cache mode ("default" for entity extraction, other for queries)
        cache_type: Type of cache (llm, query, etc.)

    Returns:
        Tuple of (content, create_time) if cache hit, None if cache miss

    Example:
        >>> cached = await handle_cache(storage, "abc123", "What is BRCA1?", "query", "llm")
        >>> if cached:
        ...     content, timestamp = cached
        ...     print(f"Cache hit: {content[:50]}...")
    """
    if hashing_kv is None:
        return None

    # Check if caching is enabled based on mode
    if mode != "default":  # handle cache for all types of query
        if not hashing_kv.global_config.get("enable_llm_cache", True):
            logger.debug("LLM cache disabled for queries")
            return None
    else:  # handle cache for entity extraction
        if not hashing_kv.global_config.get("enable_llm_cache_for_entity_extract", True):
            logger.debug("LLM cache disabled for entity extraction")
            return None

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(mode, cache_type, args_hash)

    try:
        cache_entry = await hashing_kv.get_by_id(flattened_key)
        if cache_entry:
            logger.debug(f"Cache hit: {flattened_key}")
            content = cache_entry.get("return")
            timestamp = cache_entry.get("create_time", 0)

            if content:
                logger.info(f"Cache hit [{cache_type}]: {len(content)} chars (age: {int(time.time() - timestamp)}s)")
                return content, timestamp

    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")

    logger.debug(f"Cache miss: mode={mode}, type={cache_type}")
    return None


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """Save data to cache using flattened key structure.

    Based on LightRAG save_to_cache (lightrag/utils.py:1078-1123)

    Args:
        hashing_kv: The key-value storage for caching
        cache_data: The cache data to save

    Example:
        >>> cache_data = CacheData(
        ...     args_hash="abc123",
        ...     content="BRCA1 is a tumor suppressor gene...",
        ...     prompt="What is BRCA1?",
        ...     mode="query",
        ...     cache_type="llm"
        ... )
        >>> await save_to_cache(storage, cache_data)
    """
    # Skip if storage is None or content is empty
    if hashing_kv is None or not cache_data.content:
        logger.debug("Skipping cache save: no storage or empty content")
        return

    # If content is a streaming response, don't cache it
    if hasattr(cache_data.content, "__aiter__"):
        logger.debug("Streaming response detected, skipping cache")
        return

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(cache_data.mode, cache_data.cache_type, cache_data.args_hash)

    try:
        # Check if we already have identical content cached
        existing_cache = await hashing_kv.get_by_id(flattened_key)
        if existing_cache:
            existing_content = existing_cache.get("return")
            if existing_content == cache_data.content:
                logger.debug(f"Cache duplication detected: {flattened_key}, skipping")
                return

        # Create cache entry with flattened structure
        cache_entry = {
            "return": cache_data.content,
            "cache_type": cache_data.cache_type,
            "chunk_id": cache_data.chunk_id,
            "original_prompt": cache_data.prompt,
            "queryparam": cache_data.queryparam,
            "create_time": int(time.time()),
        }

        logger.info(f"💾 Saving to cache [{cache_data.cache_type}]: {flattened_key} ({len(cache_data.content)} chars)")

        # Save using flattened key
        await hashing_kv.upsert({flattened_key: cache_entry})

    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


async def clear_cache(hashing_kv, mode: str | None = None, cache_type: str | None = None):
    """Clear cache entries matching the given filters.

    Args:
        hashing_kv: The key-value storage for caching
        mode: Optional mode filter (e.g., "query", "default")
        cache_type: Optional cache type filter (e.g., "llm", "query")

    Note:
        If both mode and cache_type are None, this function will not clear
        all caches for safety. Use clear_all_cache() for that.

    Example:
        >>> # Clear all LLM caches
        >>> await clear_cache(storage, cache_type="llm")
        >>>
        >>> # Clear all query mode caches
        >>> await clear_cache(storage, mode="query")
    """
    if hashing_kv is None:
        return

    if mode is None and cache_type is None:
        logger.warning("clear_cache called without filters - use clear_all_cache() instead")
        return

    # This is a simplified implementation
    # A full implementation would need to list all keys and filter
    logger.warning("Cache clearing not fully implemented - requires key listing support")


async def get_cache_stats(hashing_kv) -> dict[str, Any]:
    """Get statistics about cached items.

    Args:
        hashing_kv: The key-value storage for caching

    Returns:
        Dictionary with cache statistics

    Example:
        >>> stats = await get_cache_stats(storage)
        >>> print(f"Total cached items: {stats['total_items']}")
    """
    if hashing_kv is None:
        return {"total_items": 0, "error": "No cache storage"}

    # This is a simplified implementation
    # A full implementation would need storage-specific stats
    return {
        "total_items": 0,
        "note": "Cache stats not fully implemented - requires storage-specific queries",
    }
