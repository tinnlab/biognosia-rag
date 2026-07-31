"""
Reranking module for RAG query system.

Supports local and API-based reranking:
- Local: HuggingFace cross-encoder models (GPU/CPU)
- API: Jina AI, Cohere, Aliyun DashScope
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

from .api_rerankers import aliyun_rerank, cohere_rerank, jina_rerank
from .local_reranker import LocalReranker
from .processor import RerankProcessor, apply_rerank, apply_rerank_if_enabled

logger = logging.getLogger(__name__)

__all__ = [
    "LocalReranker",
    "RerankProcessor",
    "jina_rerank",
    "cohere_rerank",
    "aliyun_rerank",
    "apply_rerank",
    "apply_rerank_if_enabled",
    "create_reranker",
    "get_rerank_function",
    "create_rerank_processor",
]


def create_reranker(config: dict[str, Any]) -> object | None:
    """
    Create a reranker based on configuration.

    Args:
        config: Rerank configuration dict

    Returns:
        Reranker instance (LocalReranker) or None for API-based rerankers
    """
    provider = config.get("provider", "local")

    if provider == "local":
        # Create local reranker instance
        model = config.get("model", "BAAI/bge-reranker-v2-m3")
        device = config.get("device", "cuda:0")
        max_length = config.get("max_length", 512)
        batch_size = config.get("batch_size", 32)

        logger.info(
            f"Creating local reranker: model={model}, device={device}, max_length={max_length}, batch_size={batch_size}"
        )

        return LocalReranker(model_name=model, device=device, max_length=max_length, batch_size=batch_size)

    elif provider in ["jina", "cohere", "aliyun"]:
        # API-based rerankers don't need initialization
        logger.info(f"Using API-based reranker: {provider}")
        return None

    else:
        logger.error(f"Unknown rerank provider: {provider}")
        return None


def get_rerank_function(config: dict[str, Any]) -> Callable | None:
    """
    Get rerank function based on configuration.

    Args:
        config: Rerank configuration dict

    Returns:
        Async rerank function or None if rerank is disabled
    """
    if not config.get("enable_by_default", True):
        logger.info("Reranking disabled by default in config")
        return None

    provider = config.get("provider", "local")

    if provider == "local":
        # Create local reranker and return its rerank method
        reranker = create_reranker(config)
        if reranker:
            logger.info("Local reranker initialized successfully")
            return reranker.rerank
        else:
            logger.error("Failed to create local reranker")
            return None

    elif provider == "jina":
        # Return Jina rerank function with config
        # Fall back to JINA_API_KEY environment variable for compatibility
        import os

        api_key = config.get("api_key") or os.getenv("JINA_API_KEY")
        model = config.get("model", "jina-reranker-v2-base-multilingual")
        base_url = config.get("base_url", "https://api.jina.ai/v1/rerank")

        if not api_key:
            logger.warning("Jina API key not configured (RERANK_API_KEY or JINA_API_KEY), reranking will fail")

        async def jina_rerank_configured(query, documents, top_k=None):
            return await jina_rerank(
                query=query, documents=documents, top_k=top_k, api_key=api_key, model=model, base_url=base_url
            )

        logger.info(f"Jina reranker configured: model={model}")
        return jina_rerank_configured

    elif provider == "cohere":
        # Return Cohere rerank function with config
        # Fall back to COHERE_API_KEY environment variable for compatibility
        import os

        api_key = config.get("api_key") or os.getenv("COHERE_API_KEY")
        model = config.get("model", "rerank-multilingual-v3.0")
        base_url = config.get("base_url", "https://api.cohere.ai/v1/rerank")

        if not api_key:
            logger.warning("Cohere API key not configured (RERANK_API_KEY or COHERE_API_KEY), reranking will fail")

        async def cohere_rerank_configured(query, documents, top_k=None):
            return await cohere_rerank(
                query=query, documents=documents, top_k=top_k, api_key=api_key, model=model, base_url=base_url
            )

        logger.info(f"Cohere reranker configured: model={model}")
        return cohere_rerank_configured

    elif provider == "aliyun":
        # Return Aliyun rerank function with config
        # Fall back to ALI_API_KEY environment variable for compatibility
        import os

        api_key = config.get("api_key") or os.getenv("ALI_API_KEY")
        model = config.get("model", "gte-rerank")
        base_url = config.get(
            "base_url", "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        )

        if not api_key:
            logger.warning("Aliyun API key not configured (RERANK_API_KEY or ALI_API_KEY), reranking will fail")

        async def aliyun_rerank_configured(query, documents, top_k=None):
            return await aliyun_rerank(
                query=query, documents=documents, top_k=top_k, api_key=api_key, model=model, base_url=base_url
            )

        logger.info(f"Aliyun reranker configured: model={model}")
        return aliyun_rerank_configured

    else:
        logger.error(f"Unknown rerank provider: {provider}")
        return None


def create_rerank_processor(config: dict[str, Any]) -> RerankProcessor | None:
    """
    Create a RerankProcessor based on configuration.

    This is the recommended way to create a reranker for production use.
    It provides a clean dict-based interface compatible with both tests and production code.

    Args:
        config: Rerank configuration dict with keys:
            - enable_by_default: Whether to enable reranking (default: True)
            - provider: "local", "jina", "cohere", or "aliyun"
            - model: Model name (provider-specific)
            - device: Device for local models (e.g., "cuda:0", "cpu")
            - max_length: Max sequence length for local models
            - batch_size: Batch size for local models

    Returns:
        RerankProcessor instance or None if reranking is disabled

    Example:
        >>> rerank_config = config["reranking"]
        >>> reranker = create_rerank_processor(rerank_config)
        >>> if reranker:
        >>>     await reranker.initialize()
        >>>     global_config["rerank_processor"] = reranker
    """
    if not config.get("enable_by_default", True):
        logger.info("Reranking disabled by default in config")
        return None

    provider = config.get("provider", "local")
    logger.info(f"Creating RerankProcessor with provider: {provider}")

    return RerankProcessor(config=config)
