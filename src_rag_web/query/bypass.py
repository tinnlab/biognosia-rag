"""
Bypass query mode - direct LLM query without any retrieval.

Based on LightRAG bypass mode.
"""

import logging
from typing import Any

from .params import QueryParam

logger = logging.getLogger(__name__)


async def bypass_query(
    query: str,
    llm_provider,
    config: dict[str, Any],
    param: QueryParam | None = None,
) -> dict[str, Any]:
    """
    Bypass mode: Direct LLM query without any retrieval.

    This mode skips all retrieval steps and sends the query directly to the LLM.
    Useful for testing or when you want pure LLM generation without context.

    Args:
        query: User's query text
        llm_provider: LLM provider instance
        config: Global configuration
        param: Query parameters (optional)

    Returns:
        Dictionary with response and empty data arrays
    """
    from ..llm import generate_llm_response

    if param is None:
        param = QueryParam()

    logger.info(f"=== Bypass Query: {query[:100]}... ===")
    logger.info("Bypass mode: sending query directly to LLM without retrieval")

    try:
        # Bypass mode: send query directly to LLM without any context
        response_text = await generate_llm_response(
            prompt=query,
            llm_provider=llm_provider,
        )

        logger.info(f"Bypass query completed: {len(response_text)} chars")

        return {
            "response": response_text,
            "query": query,
            "mode": "bypass",
            "chunks": [],
            "entities": [],
            "relationships": [],
            "references": [],
            "metadata": {
                "chunks_retrieved": 0,
                "chunks_used": 0,
                "entities": 0,
                "relationships": 0,
            },
        }

    except Exception as e:
        logger.error(f"Bypass query failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        # Re-raise to fail the entire pipeline
        raise
