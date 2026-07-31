"""
Naive query mode - simple text-based search without KG traversal.

Based on LightRAG operate.py:3958-4227 (naive_query)
"""

import logging
import time
from typing import Any

from .params import QueryParam

logger = logging.getLogger(__name__)


async def naive_query(
    query: str,
    embedding_manager,
    chunks_vdb,
    text_chunks_storage,
    llm_provider,
    config: dict[str, Any],
    param: QueryParam | None = None,
    es_client: Any | None = None,
    cache_manager: Any | None = None,
) -> dict[str, Any]:
    """
    Simple text-based query without full KG traversal.

    Pipeline:
    1. Hybrid/Vector chunk retrieval (ES BM25 + Milvus semantic if enabled)
    2. Chunk processing (rerank + token limit)
    3. Context building
    4. LLM response generation

    Args:
        query: User's query text
        embedding_manager: Embedding manager
        chunks_vdb: Vector database for chunks
        text_chunks_storage: KV storage for chunk content
        llm_provider: LLM provider instance
        config: Global configuration
        param: Query parameters (optional)
        es_client: Elasticsearch client (required for hybrid search)
        cache_manager: Redis cache manager (optional, for keyword caching)

    Returns:
        Dictionary with response and metadata
    """
    from ..llm import generate_llm_response
    from ..retrieval.chunk_picking import process_chunks_unified
    from ..retrieval.context_builder import build_llm_context, build_query_context, format_references
    from ..retrieval.vector_search import get_vector_context

    if param is None:
        param = QueryParam()

    logger.info(f"=== Naive Query: {query[:100]}... ===")
    logger.info(f"Parameters: mode={param.mode}, top_k={param.chunk_top_k or 'all'}, rerank={param.enable_rerank}")

    timing = {}
    query_start = time.time()

    try:
        # Step 1: Hybrid or Vector chunk retrieval
        step_start = time.time()

        # Check if hybrid search is enabled
        use_hybrid = (
            param.enable_hybrid_search
            and es_client is not None
            and config.get("hybrid_search", {}).get("enabled", False)
        )

        if use_hybrid:
            from ..retrieval.elasticsearch_search import get_hybrid_context
            from ..retrieval.keyword_expansion import expand_query_keywords

            logger.debug("Step 1: Hybrid search (ES BM25 + Milvus semantic)")

            # Keyword expansion
            keyword_data = await expand_query_keywords(
                query=query,
                llm_provider=llm_provider,
                config=config,
                cache_manager=cache_manager,
            )
            keywords = keyword_data["keyword_string"]
            logger.info(
                f"Keyword expansion: {len(keyword_data.get('all_keywords', []))} keywords "
                f"({'cached' if keyword_data.get('cached') else 'fresh'})"
            )

            # Hybrid search
            hybrid_result = await get_hybrid_context(
                query=query,
                keywords=keywords,
                es_client=es_client,
                chunks_vdb=chunks_vdb,
                text_chunks_storage=text_chunks_storage,
                embedding_manager=embedding_manager,
                llm_provider=llm_provider,
                config=config,
                query_param=param,
            )

            vector_chunks = hybrid_result["chunks"]
            timing["hybrid_search"] = time.time() - step_start
            logger.info(
                f"Hybrid search retrieved {len(vector_chunks)} chunks "
                f"(ES={hybrid_result['es_count']}, Milvus={hybrid_result['milvus_count']}, "
                f"Overlap={hybrid_result['overlap_count']}) ({timing['hybrid_search']:.2f}s)"
            )
        else:
            logger.debug("Step 1: Vector search for chunks")
            vector_chunks = await get_vector_context(
                query=query,
                embedding_manager=embedding_manager,
                chunks_vdb=chunks_vdb,
                text_chunks_storage=text_chunks_storage,
                top_k=param.chunk_top_k or 100,  # Retrieve more initially
                cosine_threshold=param.cosine_threshold,
                llm_provider=llm_provider,
                enable_query_expansion=config.get("ENABLE_CANDIDATE_FILTERING", True),
                num_query_expansions=config.get("NUM_QUERY_EXPANSIONS", 5),
                max_tokens_query_expansion=config.get("MAX_TOKENS_QUERY_EXPANSION", 5000),
            )
            timing["vector_search"] = time.time() - step_start
            logger.info(f"Vector search retrieved {len(vector_chunks)} chunks ({timing['vector_search']:.2f}s)")

        if not vector_chunks:
            logger.warning("No chunks retrieved from vector search")
            return {
                "response": "I don't have enough information to answer this question based on the available data.",
                "query": query,
                "mode": param.mode,
                "chunks": [],
                "references": [],
                "metadata": {
                    "chunks_retrieved": 0,
                    "chunks_used": 0,
                },
            }

        # Step 2: Process chunks (rerank + token truncation + ID assignment)
        logger.debug("Step 2: Processing chunks")
        step_start = time.time()

        # Create a global_config dict for process_chunks_unified
        # Prefer rerank_processor (new) over rerank_model_func (legacy)
        global_config = {
            "tokenizer": config.get("tokenizer"),
            "MAX_TOTAL_TOKENS": param.max_total_tokens,
            "rerank_processor": config.get("rerank_processor"),  # New: RerankProcessor instance
            "rerank_model_func": config.get("rerank_model_func"),  # Legacy fallback
            "min_rerank_score": param.min_rerank_score,
        }

        processed_chunks = await process_chunks_unified(
            query=query,
            unique_chunks=vector_chunks,
            query_param=param,
            global_config=global_config,
            source_type="vector",
            chunk_token_limit=param.max_total_tokens,
        )
        timing["chunk_processing"] = time.time() - step_start
        logger.info(f"Processed to {len(processed_chunks)} chunks ({timing['chunk_processing']:.2f}s)")

        # Step 3: Build context
        logger.debug("Step 3: Building context")
        step_start = time.time()
        context = build_query_context(
            chunks=processed_chunks,
            entities=[],
            relationships=[],
        )
        timing["context_building"] = time.time() - step_start

        # Step 4: Format for LLM
        logger.debug("Step 4: Formatting LLM prompt")
        llm_prompt = build_llm_context(
            query=query,
            context=context,
            mode="naive",
            response_type=param.response_type,
        )

        # Step 5: Generate LLM response
        logger.debug("Step 5: Generating LLM response")
        step_start = time.time()
        response_text = await generate_llm_response(
            prompt=llm_prompt,
            llm_provider=llm_provider,
        )
        timing["llm_generation"] = time.time() - step_start

        # Step 6: Format references
        references = format_references(processed_chunks)

        timing["total"] = time.time() - query_start

        logger.info(f"Naive query completed: {len(response_text)} chars, {len(references)} references")
        logger.info(
            f"Timing: total={timing['total']:.2f}s | vector={timing.get('vector_search', 0):.2f}s | "
            f"process={timing.get('chunk_processing', 0):.2f}s | llm={timing.get('llm_generation', 0):.2f}s"
        )

        return {
            "response": response_text,
            "query": query,
            "mode": "naive",
            "chunks": processed_chunks,
            "references": references,
            "metadata": {
                "chunks_retrieved": len(vector_chunks),
                "chunks_used": len(processed_chunks),
            },
        }

    except Exception as e:
        logger.error(f"Naive query failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        # Re-raise to fail the entire pipeline
        raise
