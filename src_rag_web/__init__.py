"""
RAG Query System

A lightweight, production-ready RAG query system extracted from LightRAG.
This package provides query-only functionality for knowledge graph-enhanced retrieval.
"""

from __future__ import annotations

import logging
from typing import Optional

__version__ = "0.1.0"
__author__ = "Biognosia RAG Team"

logger = logging.getLogger(__name__)

# Convenience re-exports. These pull in the heavy retrieval/embedding/storage
# stack (torch, transformers, pymilvus, nltk, ...). They are optional: the slim
# test image (MCP_TEST_MODE) does not install those extras and the MCP server
# never uses these re-exports, so we degrade gracefully rather than fail to
# import the package. In the full image every dependency is present and the
# names below are exported normally.
EntityStatisticsManager = None  # type: ignore[assignment]
try:
    from .entity_extraction import EntityStatisticsManager
    from .query import QueryParam, bypass_query, kg_query, naive_query
    from .storage.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
    from .storage.milvus_storage import MilvusStorage
    from .storage.neo4j_storage import Neo4jStorage
    from .storage.redis_storage import RedisStorage
except ImportError as exc:  # only hit in the slim test image
    logger.debug("Optional RAG extras unavailable (%s); package re-exports disabled", exc)

# Global entity statistics manager (singleton)
_entity_statistics_manager = None


def initialize_entity_statistics(config: dict) -> EntityStatisticsManager | None:
    """
    Initialize entity statistics manager at application startup.

    Should be called once when application starts, after loading configuration.

    Args:
        config: Configuration dictionary with ngram_entity_matching settings

    Returns:
        EntityStatisticsManager instance if configured, None otherwise

    Example:
        >>> from src_rag import initialize_entity_statistics
        >>> from src_rag.config import load_config
        >>>
        >>> config = load_config('config/rag.conf')
        >>> stats_mgr = initialize_entity_statistics(config)
    """
    global _entity_statistics_manager

    ngram_config = config.get("ngram_entity_matching", {})

    if not ngram_config.get("enable", False):
        logger.info("N-gram entity matching disabled")
        return None

    stats_file = ngram_config.get("statistics_file")
    if not stats_file:
        logger.warning("N-gram matching enabled but no statistics file configured")
        return None

    logger.info("Initializing entity statistics manager...")
    _entity_statistics_manager = EntityStatisticsManager(stats_file)
    _entity_statistics_manager.load_statistics()

    # Store in config for query pipeline access
    config["entity_stats_manager"] = _entity_statistics_manager

    logger.info("Entity statistics manager initialized successfully")
    return _entity_statistics_manager


def get_entity_statistics_manager() -> EntityStatisticsManager | None:
    """
    Get the global entity statistics manager.

    Returns:
        EntityStatisticsManager instance if initialized, None otherwise
    """
    return _entity_statistics_manager


__all__ = [
    # Storage
    "BaseVectorStorage",
    "BaseKVStorage",
    "BaseGraphStorage",
    "MilvusStorage",
    "RedisStorage",
    "Neo4jStorage",
    # Query
    "QueryParam",
    "naive_query",
    "kg_query",
    "bypass_query",
    # Entity extraction
    "EntityStatisticsManager",
    "initialize_entity_statistics",
    "get_entity_statistics_manager",
]
