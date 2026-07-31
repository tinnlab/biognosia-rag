"""
API-based rerankers (Jina, Cohere, Aliyun).

Adapted from plans/lightrag-code/rerank/rerankers.py
"""

import logging

import aiohttp

logger = logging.getLogger(__name__)


async def jina_rerank(
    query: str,
    documents: list[str],
    top_k: int | None = None,
    api_key: str | None = None,
    model: str = "jina-reranker-v2-base-multilingual",
    base_url: str = "https://api.jina.ai/v1/rerank",
) -> list[tuple[int, float]]:
    """
    Rerank documents using Jina AI reranker.

    Args:
        query: The search query
        documents: List of document texts to rerank
        top_k: Number of top results to return
        api_key: Jina API key (required)
        model: Model name
        base_url: API base URL

    Returns:
        List of tuples (document_index, relevance_score) sorted by relevance
    """
    if not api_key:
        logger.error("Jina API key is required")
        return [(i, 0.0) for i in range(min(top_k or len(documents), len(documents)))]

    if not documents:
        return []

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_k or len(documents),
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Jina rerank API error: {response.status} - {error_text}")
                    return [(i, 0.0) for i in range(min(top_k or len(documents), len(documents)))]

                result = await response.json()
                results = result.get("results", [])

                # Extract (index, score) pairs
                reranked = [(item["index"], item["relevance_score"]) for item in results]

                logger.debug(f"Jina reranked {len(documents)} documents, returning top {len(reranked)}")

                return reranked

    except Exception as e:
        logger.error(f"Error during Jina reranking: {str(e)}")
        return [(i, 0.0) for i in range(min(top_k or len(documents), len(documents)))]


async def cohere_rerank(
    query: str,
    documents: list[str],
    top_k: int | None = None,
    api_key: str | None = None,
    model: str = "rerank-multilingual-v3.0",
    base_url: str = "https://api.cohere.ai/v1/rerank",
) -> list[tuple[int, float]]:
    """
    Rerank documents using Cohere reranker.

    Args:
        query: The search query
        documents: List of document texts to rerank
        top_k: Number of top results to return
        api_key: Cohere API key (required)
        model: Model name (rerank-v3.5, rerank-multilingual-v3.0)
        base_url: API base URL

    Returns:
        List of tuples (document_index, relevance_score) sorted by relevance
    """
    if not api_key:
        logger.error("Cohere API key is required")
        return [(i, 0.0) for i in range(min(top_k or len(documents), len(documents)))]

    if not documents:
        return []

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_k or len(documents),
        "return_documents": False,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Cohere rerank API error: {response.status} - {error_text}")
                    return [(i, 0.0) for i in range(min(top_k or len(documents), len(documents)))]

                result = await response.json()
                results = result.get("results", [])

                # Extract (index, score) pairs
                reranked = [(item["index"], item["relevance_score"]) for item in results]

                logger.debug(f"Cohere reranked {len(documents)} documents, returning top {len(reranked)}")

                return reranked

    except Exception as e:
        logger.error(f"Error during Cohere reranking: {str(e)}")
        return [(i, 0.0) for i in range(min(top_k or len(documents), len(documents)))]


async def aliyun_rerank(
    query: str,
    documents: list[str],
    top_k: int | None = None,
    api_key: str | None = None,
    model: str = "gte-rerank",
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
) -> list[tuple[int, float]]:
    """
    Rerank documents using Alibaba Cloud (Aliyun) DashScope reranker.

    Args:
        query: The search query
        documents: List of document texts to rerank
        top_k: Number of top results to return
        api_key: Aliyun API key (required)
        model: Model name (gte-rerank, gte-rerank-hybrid)
        base_url: API base URL

    Returns:
        List of tuples (document_index, relevance_score) sorted by relevance
    """
    if not api_key:
        logger.error("Aliyun API key is required")
        return [(i, 0.0) for i in range(min(top_k or len(documents), len(documents)))]

    if not documents:
        return []

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Format documents for Aliyun API
    ali_documents = [{"text": doc} for doc in documents]

    data = {
        "model": model,
        "input": {"query": query, "documents": ali_documents},
        "parameters": {"top_n": top_k or len(documents), "return_documents": False},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Aliyun rerank API error: {response.status} - {error_text}")
                    return [(i, 0.0) for i in range(min(top_k or len(documents), len(documents)))]

                result = await response.json()
                results = result.get("output", {}).get("results", [])

                # Extract (index, score) pairs
                reranked = [(item["index"], item["relevance_score"]) for item in results]

                logger.debug(f"Aliyun reranked {len(documents)} documents, returning top {len(reranked)}")

                return reranked

    except Exception as e:
        logger.error(f"Error during Aliyun reranking: {str(e)}")
        return [(i, 0.0) for i in range(min(top_k or len(documents), len(documents)))]
