"""
Auto mode query with LLM function calling for intelligent routing.

Uses validated tool schemas and system prompt to let the LLM decide:
1. No tool call -> Answer directly (bypass mode)
2. retrieve_entities -> Extract entities only (entities_only mode)
3. retrieve_full_context -> Full RAG with entities + papers (mix mode)

Validation Results:
- Baseline prompt: 96.3% factoids, 98.9% multi-paper
- Model: Llama 4 Scout (17B) - meta-llama/llama-4-scout-17b-16e-instruct
- Prompt contributes 99% to routing accuracy
"""

import logging
from typing import Any

from .params import QueryParam

logger = logging.getLogger(__name__)

# Validated tool schemas (96.3%/98.9% accuracy with Llama 4 Scout)
RETRIEVE_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_entities",
        "description": (
            "Get biomedical entity information from curated databases. "
            "Call this when you need entity definitions, properties, "
            "or to identify what pathway/disease genes belong to."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question"
                }
            },
            "required": ["query"]
        }
    }
}

RETRIEVE_FULL_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_full_context",
        "description": (
            "Get comprehensive information including entities AND research papers. "
            "Call this when you need evidence, mechanisms, comparisons, or experimental details. "
            "Includes everything from retrieve_entities plus papers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question"
                }
            },
            "required": ["query"]
        }
    }
}

TOOLS = [RETRIEVE_ENTITIES_TOOL, RETRIEVE_FULL_CONTEXT_TOOL]

# Validated system prompt (updated for OSS 120B compatibility)
# Original: 96.3%/98.9% accuracy with Llama 4 Scout (17B)
# Updated: Works with both Llama 4 Scout and OSS 120B
AUTO_MODE_SYSTEM_PROMPT = """You are a biomedical assistant. Use tools to retrieve research information.

retrieve_entities - Use for gene set and entity questions:
- "What pathway/disease/process represents gene set: GENE1, GENE2..."
- "Which pathway do these genes belong to: GENE1, GENE2..."
- "What is the biological function of genes in [pathway name]?"
- Entity definitions and classifications
- Identifying biological functions from gene lists

retrieve_full_context - Use ONLY when you need research papers:
- Experimental results, measurements, statistics (copy numbers, RMSD, expression levels)
- Detailed mechanisms from specific studies
- Comparisons requiring paper citations
- Questions explicitly asking for evidence or experimental details

Direct answer - NEVER use for questions with gene lists or specific entities.

CRITICAL: If the question lists genes (GENE1, GENE2, etc.) and asks what pathway/disease/process
they represent, you MUST call retrieve_entities. Do NOT answer directly."""


async def query_with_tools(
    query: str,
    llm_provider,
    config: dict[str, Any],
    embedding_manager,
    storage_dict: dict[str, Any],
    param: QueryParam | None = None,
    es_client=None,
    cache_manager=None,
    worker_pool=None,
    rerank_processor=None,
    detailed_logger=None,
) -> dict[str, Any]:
    """
    Auto mode: Let LLM decide what to retrieve using function calling.

    The LLM chooses between:
    1. No tool call -> Bypass mode (direct answer)
    2. retrieve_entities -> Entities-only mode (no papers)
    3. retrieve_full_context -> Mix mode (entities + papers)

    Smart deduplication: If both tools called, use retrieve_full_context
    (it includes everything from retrieve_entities plus papers).

    Args:
        query: User's query text
        llm_provider: LLM provider instance (must support generate_with_tools)
        config: Global configuration
        embedding_manager: Embedding manager instance
        storage_dict: Storage instances
        param: Query parameters (optional)
        es_client: Elasticsearch client (for mix mode)
        cache_manager: Redis cache manager
        worker_pool: Retrieval worker pool (for mix mode)
        rerank_processor: Reranking processor
        detailed_logger: Detailed logging instance

    Returns:
        Dictionary with response and retrieved data
    """
    from .kg.mix import mix_mode

    if param is None:
        param = QueryParam()

    logger.info(f"=== Auto Mode Query: {query[:100]}... ===")
    logger.info("Using LLM function calling to determine retrieval strategy")

    try:
        # Check if provider supports function calling
        if not hasattr(llm_provider, "generate_with_tools"):
            logger.warning(
                f"Provider {llm_provider.__class__.__name__} does not support function calling. "
                "Falling back to mix mode."
            )
            return await mix_mode(
                query=query,
                embedding_manager=embedding_manager,
                storage_dict=storage_dict,
                llm_provider=llm_provider,
                config=config,
                param=param,
                es_client=es_client,
                cache_manager=cache_manager,
                worker_pool=worker_pool,
                detailed_logger=detailed_logger,
            )

        # Call LLM with tools for routing decision
        logger.info("Calling LLM with tool schemas for routing decision...")
        logger.debug(f"System prompt ({len(AUTO_MODE_SYSTEM_PROMPT)} chars):\n{AUTO_MODE_SYSTEM_PROMPT}")
        logger.debug(f"User query ({len(query)} chars): {query[:200]}...")
        logger.debug(f"Tools: {[t['function']['name'] for t in TOOLS]}")

        response = await llm_provider.generate_with_tools(
            prompt=query,
            system_prompt=AUTO_MODE_SYSTEM_PROMPT,
            tools=TOOLS,
        )

        # Parse tool calls
        tool_calls = response.get("tool_calls", [])
        response_text = response.get("content", "")

        # Log routing decision with full details
        logger.info(f"LLM response - tool_calls: {tool_calls}")
        logger.info(f"LLM response - content: {response_text[:200] if response_text else None}...")
        logger.info(f"LLM response - finish_reason: {response.get('finish_reason')}")

        if tool_calls:
            tool_names = [tc.get("name") for tc in tool_calls]
            logger.info(f"LLM decided to call tools: {tool_names}")
        else:
            logger.info("LLM decided to answer directly (bypass mode)")

        # Execute based on tool decision
        return await _execute_tool_calls(
            query=query,
            tool_calls=tool_calls,
            response_text=response_text,
            llm_provider=llm_provider,
            config=config,
            embedding_manager=embedding_manager,
            storage_dict=storage_dict,
            param=param,
            es_client=es_client,
            cache_manager=cache_manager,
            worker_pool=worker_pool,
            rerank_processor=rerank_processor,
            detailed_logger=detailed_logger,
        )

    except Exception as e:
        logger.error(f"Auto mode query failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Fallback to mix mode on error
        logger.warning("Falling back to mix mode due to error")
        return await mix_mode(
            query=query,
            embedding_manager=embedding_manager,
            storage_dict=storage_dict,
            llm_provider=llm_provider,
            config=config,
            param=param,
            es_client=es_client,
            cache_manager=cache_manager,
            worker_pool=worker_pool,
            detailed_logger=detailed_logger,
        )


async def _execute_tool_calls(
    query: str,
    tool_calls: list[dict],
    response_text: str,
    llm_provider,
    config: dict[str, Any],
    embedding_manager,
    storage_dict: dict[str, Any],
    param: QueryParam,
    es_client,
    cache_manager,
    worker_pool,
    rerank_processor,
    detailed_logger,
) -> dict[str, Any]:
    """
    Execute tool calls based on LLM decision.

    Smart deduplication: If both tools called, use retrieve_full_context only
    (it includes everything from retrieve_entities plus papers).

    Args:
        tool_calls: List of tool calls from LLM
        response_text: Direct response text (if no tool calls)
        (other args same as query_with_tools)

    Returns:
        Query result dictionary
    """
    from .bypass import bypass_query
    from .kg.mix import mix_mode

    # Case 1: No tool calls -> Bypass mode (direct answer)
    if not tool_calls:
        logger.info("No tool calls -> using bypass mode")
        # We already have the response from the routing call - use it directly!
        if response_text:
            logger.info(f"Using response from routing call (length: {len(response_text)} chars)")
            return {
                "response": response_text,
                "mode": "auto:bypass",
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            }
        else:
            # No response text - call bypass_query as fallback
            logger.warning("No response text from routing call, calling bypass_query...")
            return await bypass_query(
                query=query,
                llm_provider=llm_provider,
                config=config,
                param=param,
            )

    # Case 2: Smart deduplication - if both tools called, use full_context only
    tool_names = [tc.get("name") for tc in tool_calls]

    if "retrieve_full_context" in tool_names:
        # retrieve_full_context includes entities, so use it exclusively
        logger.info("retrieve_full_context called -> using mix mode (includes entities)")
        result = await mix_mode(
            query=query,
            embedding_manager=embedding_manager,
            storage_dict=storage_dict,
            llm_provider=llm_provider,
            config=config,
            param=param,
            es_client=es_client,
            cache_manager=cache_manager,
            worker_pool=worker_pool,
            detailed_logger=detailed_logger,
        )
        result["mode"] = "auto:retrieve_full_context"
        return result

    elif "retrieve_entities" in tool_names:
        # retrieve_entities only (no papers) - use mix mode with entities_only flag
        logger.info("retrieve_entities called -> using mix mode with entities_only=True")
        result = await mix_mode(
            query=query,
            embedding_manager=embedding_manager,
            storage_dict=storage_dict,
            llm_provider=llm_provider,
            config=config,
            param=param,
            es_client=es_client,
            cache_manager=cache_manager,
            worker_pool=worker_pool,
            detailed_logger=detailed_logger,
            entities_only=True,  # Skip Neo4j and chunks
        )
        result["mode"] = "auto:retrieve_entities"
        return result

    else:
        # Unknown tool or empty tool_calls -> fallback to mix mode
        logger.warning(f"Unknown tools called: {tool_names}, falling back to mix mode")
        result = await mix_mode(
            query=query,
            embedding_manager=embedding_manager,
            storage_dict=storage_dict,
            llm_provider=llm_provider,
            config=config,
            param=param,
            es_client=es_client,
            cache_manager=cache_manager,
            worker_pool=worker_pool,
            detailed_logger=detailed_logger,
        )
        result["mode"] = "auto:fallback_mix"
        return result
