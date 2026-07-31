"""
Keyword extraction functions for RAG query system.

Based on LightRAG operate.py:2446-2585
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def get_keywords_from_query(
    query: str,
    query_param,  # QueryParam
    llm_provider,
    global_config: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Based on LightRAG operate.py:2446-2476

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        llm_provider: LLM provider for keyword extraction
        global_config: Global configuration dictionary

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if hasattr(query_param, "hl_keywords") and query_param.hl_keywords:
        hl_keywords = query_param.hl_keywords
        ll_keywords = query_param.ll_keywords if hasattr(query_param, "ll_keywords") else []
        return hl_keywords, ll_keywords

    # Extract keywords using extract_keywords_only function
    hl_keywords, ll_keywords = await extract_keywords_only(query, query_param, llm_provider, global_config)
    return hl_keywords, ll_keywords


async def extract_keywords_only(
    text: str,
    param,  # QueryParam
    llm_provider,
    global_config: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).

    Based on LightRAG operate.py:2478-2585

    Args:
        text: Query text to extract keywords from
        param: Query parameters
        llm_provider: LLM provider for extraction
        global_config: Global configuration (optional)

    Returns:
        Tuple of (high_level_keywords, low_level_keywords)
    """
    from ..utils.helpers import remove_think_tags
    from .prompts import PROMPTS

    # Build the examples (join all example strings)
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])

    # Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        examples=examples,
        query=text,
    )

    logger.debug(f"[extract_keywords] Extracting keywords from query: {text[:100]}...")

    try:
        # Call the LLM for keyword extraction
        result = await llm_provider.generate(kw_prompt)

        # Remove think tags if present
        result = remove_think_tags(result)

        # Parse out JSON from the LLM response
        # Try to find JSON in the response
        result = result.strip()

        # Remove markdown code fences if present
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        try:
            keywords_data = json.loads(result)
            if not keywords_data:
                logger.error("No JSON-like structure found in the LLM response.")
                return [], []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"LLM response: {result}")
            # Try to use json_repair if available
            try:
                import json_repair

                keywords_data = json_repair.loads(result)
                # Check if json_repair returned valid dict
                if not isinstance(keywords_data, dict):
                    logger.error(f"json_repair returned non-dict: {type(keywords_data)}")
                    return [], []
            except (ImportError, Exception) as repair_error:
                logger.error(f"json_repair also failed: {repair_error}")
                return [], []

        # Verify keywords_data is a dict before accessing
        if not isinstance(keywords_data, dict):
            logger.error(f"keywords_data is not a dict: {type(keywords_data)}")
            return [], []

        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])

        logger.info(f"Extracted keywords - HL: {hl_keywords}, LL: {ll_keywords}")

        return hl_keywords, ll_keywords

    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return [], []
