"""
Knowledge graph query module.

Routes to appropriate mode (local, global, hybrid, mix).
"""

import logging
import time
from datetime import datetime
from typing import Any

from ...debug_logging.detailed_logger import DetailedLogger
from ..params import QueryParam
from .global_mode import global_mode
from .hybrid import hybrid_mode
from .local import local_mode
from .mix import mix_mode

logger = logging.getLogger(__name__)


async def kg_query(
    query: str,
    embedding_manager,
    storage_dict: dict[str, Any],
    llm_provider,
    config: dict[str, Any],
    param: QueryParam | None = None,
    es_client: Any | None = None,
    cache_manager: Any | None = None,
    worker_pool: Any | None = None,
) -> dict[str, Any]:
    """
    Knowledge graph-enhanced query supporting multiple modes.

    Modes:
    - local: Entity-focused (entities -> relationships -> chunks)
    - global: Relationship-focused (relationships -> entities -> chunks)
    - hybrid: Combines local + global results
    - mix: KG data + vector chunks combined (uses persistent worker pool)

    Args:
        query: User's query text
        embedding_manager: Embedding manager
        storage_dict: Dictionary containing all storage instances
        llm_provider: LLM provider instance
        config: Global configuration
        param: Query parameters
        es_client: Elasticsearch client (optional, for hybrid search)
        cache_manager: Redis cache manager (optional)
        worker_pool: RetrievalWorkerPool instance (required for mix mode)

    Returns:
        Dictionary with response and metadata
    """
    if param is None:
        param = QueryParam()

    logger.info(f"=== KG Query: {query[:100]}... ===")
    logger.info(f"Mode: {param.mode}, top_k={param.top_k}")

    # Initialize detailed logger
    detailed_logger = DetailedLogger(log_dir=param.log_dir, query_id=param.query_id)

    # Log metadata
    detailed_logger.log_metadata(
        {
            "query_id": param.query_id,
            "query": query,
            "mode": param.mode,
            "top_k": param.top_k,
            "chunk_top_k": param.chunk_top_k,
            "enable_rerank": param.enable_rerank,
            "timestamp": datetime.now().isoformat(),
        }
    )

    query_start = time.time()

    try:
        # Route to appropriate mode
        if param.mode == "local":
            result = await local_mode(
                query, embedding_manager, storage_dict, llm_provider, config, param, es_client, cache_manager
            )
        elif param.mode == "global":
            result = await global_mode(
                query, embedding_manager, storage_dict, llm_provider, config, param, es_client, cache_manager
            )
        elif param.mode == "hybrid":
            result = await hybrid_mode(
                query, embedding_manager, storage_dict, llm_provider, config, param, es_client, cache_manager
            )
        elif param.mode == "mix":
            result = await mix_mode(
                query,
                embedding_manager,
                storage_dict,
                llm_provider,
                config,
                param,
                es_client,
                cache_manager,
                worker_pool,
                detailed_logger,
            )
        else:
            error_msg = f"Unknown query mode: {param.mode}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Add timing
        result["timing"] = {"total": time.time() - query_start}

        # Flush detailed logs to disk
        if detailed_logger:
            detailed_logger.flush()
            logger.debug("Detailed logs flushed to disk")

        return result

    except Exception as e:
        logger.error(f"KG query failed: {e}")

        # Flush logs even on error
        if "detailed_logger" in locals() and detailed_logger:
            try:
                detailed_logger.flush()
            except Exception as flush_error:
                logger.error(f"Failed to flush detailed logs: {flush_error}")

        import traceback

        logger.error(traceback.format_exc())
        # Re-raise to fail the entire pipeline
        raise


__all__ = ["kg_query"]
