"""
Entities-only retrieval mode.

Extracts biomedical entities (genes, diseases, chemicals, etc.) without
retrieving full research papers. Useful for queries that can be answered
with structured entity information alone.

Pipeline:
1. Phase 1: N-gram entity matching
2. Phase 2: Semantic community discovery
3. Neo4j expansion: DISABLED (always)
4. Entity reranking (if enabled)
5. LLM response with entity context only (NO CHUNKS, NO RELATIONSHIPS)
"""

import logging
from typing import Any

from .params import QueryParam

logger = logging.getLogger(__name__)


async def entities_only_query(
    query: str,
    llm_provider,
    config: dict[str, Any],
    ngram_matcher,
    semantic_community,
    redis_client,
    param: QueryParam | None = None,
) -> dict[str, Any]:
    """
    Entities-only mode: Extract entities without retrieving papers.

    This mode extracts relevant biomedical entities using both n-gram matching
    and semantic community discovery, but does NOT retrieve research papers,
    relationships, or perform Neo4j expansion.

    Args:
        query: User's query text
        llm_provider: LLM provider instance
        config: Global configuration
        ngram_matcher: N-gram matcher for stage 1 entity extraction
        semantic_community: Semantic community discoverer for stage 2
        redis_client: Redis client for entity data lookup
        param: Query parameters (optional)

    Returns:
        Dictionary with response and entity data (no chunks/relationships)
    """
    from ..llm import generate_llm_response
    from ..llm.prompts import PROMPTS

    if param is None:
        param = QueryParam()

    logger.info(f"=== Entities-Only Query: {query[:100]}... ===")
    logger.info("Entities-only mode: extracting entities without papers")

    try:
        # Phase 1: N-gram entity extraction
        logger.info("Phase 1: N-gram entity matching...")
        stage1_entities = ngram_matcher.match_entities(query)
        logger.info(f"Found {len(stage1_entities)} stage 1 entities")

        # Phase 2: Semantic community discovery
        logger.info("Phase 2: Semantic community discovery...")
        stage2_entities = await semantic_community.discover_entities(
            query=query,
            existing_entities=stage1_entities,
            config=config
        )
        logger.info(f"Found {len(stage2_entities)} stage 2 entities")

        # Combine entities
        all_entities = stage1_entities + stage2_entities

        # Optional: Entity reranking (if enabled in config)
        if config.get("reranking", {}).get("enable_entity_reranking", False):
            logger.info("Reranking entities...")
            from ..reranking import rerank_entities
            all_entities = await rerank_entities(
                query=query,
                entities=all_entities,
                config=config
            )

        # Limit entities to top_k
        top_k_entities = param.top_k_entities or config.get("query", {}).get(
            "top_k_entities", 30
        )
        entities = all_entities[:top_k_entities]

        logger.info(f"Using top {len(entities)} entities for response")

        # Fetch entity details from Redis
        entity_data = []
        for entity in entities:
            # Fetch entity metadata from Redis
            entity_key = f"lightrag_entities:{entity['id']}"
            entity_info = redis_client.get(entity_key)

            if entity_info:
                import json
                entity_info = json.loads(entity_info)
                entity_data.append({
                    "name": entity.get("name", entity.get("id")),
                    "type": entity.get("type", "Unknown"),
                    "description": entity_info.get("description", ""),
                    "degree": entity_info.get("degree", 0),
                    "source": entity.get("source", "unknown"),
                    "score": entity.get("score", 0.0)
                })
            else:
                # Fallback if Redis data not available
                entity_data.append({
                    "name": entity.get("name", entity.get("id")),
                    "type": entity.get("type", "Unknown"),
                    "description": "",
                    "degree": 0,
                    "source": entity.get("source", "unknown"),
                    "score": entity.get("score", 0.0)
                })

        # Format entities for LLM prompt
        entity_text = _format_entities_for_prompt(entity_data)

        # Get entities_only prompt template
        prompt_template = PROMPTS.get("entities_only_prompt", "")

        # Fill in template
        full_prompt = prompt_template.format(
            entity_data=entity_text,
            query=query
        )

        # Generate response
        logger.info("Generating LLM response with entity context...")
        response_text = await generate_llm_response(
            prompt=full_prompt,
            llm_provider=llm_provider,
        )

        logger.info(f"Entities-only query completed: {len(response_text)} chars")

        return {
            "response": response_text,
            "query": query,
            "mode": "entities_only",
            "chunks": [],  # No chunks in entities-only mode
            "entities": entity_data,
            "relationships": [],  # No relationships in entities-only mode
            "references": [],
            "metadata": {
                "chunks_retrieved": 0,
                "chunks_used": 0,
                "entities": len(entity_data),
                "entities_stage1": len(stage1_entities),
                "entities_stage2": len(stage2_entities),
                "relationships": 0,
                "neo4j_expansion": False,  # Always disabled
            },
        }

    except Exception as e:
        logger.error(f"Entities-only query failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def _format_entities_for_prompt(entities: list[dict]) -> str:
    """
    Format entities as readable text for LLM prompt.

    Args:
        entities: List of entity dictionaries

    Returns:
        Formatted string with entity information
    """
    if not entities:
        return "No entities found."

    lines = []
    for entity in entities:
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")
        description = entity.get("description", "")
        degree = entity.get("degree", 0)

        # Format: "- BRCA1 (Gene, degree: 245): DNA repair protein..."
        if description:
            lines.append(f"- {name} ({entity_type}, degree: {degree}): {description}")
        else:
            lines.append(f"- {name} ({entity_type}, degree: {degree})")

    return "\n".join(lines)
