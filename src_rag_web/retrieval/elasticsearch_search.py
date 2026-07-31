"""
Elasticsearch full text search integration for RAG.

Combines BM25 keyword search with semantic vector search using
Reciprocal Rank Fusion (RRF) for hybrid retrieval.
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def search_elasticsearch(
    keywords: str,
    es_client: Any,
    index_name: str = "lightrag_chunks",
    top_k: int = 50,
    fuzziness: str = "AUTO",
    operator: str = "or",
    detailed_logger: Optional[Any] = None,
) -> list[dict]:
    """
    Search Elasticsearch with BM25 ranking.

    Uses simple_query_string to avoid clause limit issues with many keywords.

    Args:
        keywords: Space-separated keywords (from keyword expansion)
        es_client: Elasticsearch async client
        index_name: Index to search
        top_k: Number of results to return
        fuzziness: Fuzzy matching level ("AUTO", "0", "1", "2")
        operator: Query operator ("or" or "and")
        detailed_logger: Optional DetailedLogger for structured logging

    Returns:
        List of chunk results with BM25 scores
    """
    start_time = time.perf_counter()

    try:
        # Use single match query on content field for efficiency
        # This avoids maxClauseCount issues with multi_match + fuzziness
        # BM25 handles multiple keywords naturally in a single clause
        # Disable fuzziness to avoid clause explosion (each fuzzy term creates many clauses)
        fuzz_setting = "0" if fuzziness == "AUTO" else fuzziness

        query = {
            "match": {
                "content": {
                    "query": keywords,
                    "operator": operator,
                    "fuzziness": fuzz_setting,
                    "minimum_should_match": "30%",  # At least 30% of terms must match
                }
            }
        }

        # Log ES query for debugging
        import json as json_module

        logger.info(f"Elasticsearch query: {json_module.dumps(query, indent=2)}")
        logger.info(f"Keywords ({len(keywords.split())} terms): {keywords[:200]}...")

        response = await es_client.search(
            index=index_name,
            query=query,
            size=top_k,
            _source=["full_doc_id", "chunk_order_index"],  # Don't fetch content - will be fetched later from Redis
            explain=False,
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append(
                {
                    "id": hit["_id"],  # Use 'id' for consistency with vector/KG search
                    "chunk_id": hit["_id"],  # Keep for backward compatibility
                    "score": hit["_score"],
                    "full_doc_id": hit["_source"].get("full_doc_id", ""),
                    "chunk_order_index": hit["_source"].get("chunk_order_index", 0),
                    "source": "elasticsearch",
                    "rank": len(results) + 1,
                }
            )

        # Calculate timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Elasticsearch search: {len(results)} results (top score: {results[0]['score']:.2f})"
            if results
            else "Elasticsearch search: 0 results"
        )

        # Log top results for debugging
        if results:
            logger.info("ES top 10 results:")
            for i, r in enumerate(results[:10], 1):
                logger.info(f"  {i}. {r['chunk_id']} (score={r['score']:.2f})")

        # Detailed logging if logger provided
        if detailed_logger:
            # Log all chunks to JSONL
            for result in results:
                detailed_logger.log_retrieval_elasticsearch_chunk(
                    {"chunk_id": result["chunk_id"], "score": result["score"], "rank": result["rank"]}
                )

            # Log summary
            num_keywords = len(keywords.split())
            top_score = results[0]["score"] if results else 0.0
            detailed_logger.log_retrieval_elasticsearch_summary(
                {
                    "num_keywords": num_keywords,
                    "total_results": len(results),
                    "top_score": top_score,
                    "timing_ms": elapsed_ms,
                }
            )

        return results

    except Exception as e:
        logger.error(f"Elasticsearch search failed: {e}")
        return []


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(doc) = sum(1 / (k + rank(doc)))

    Args:
        result_lists: List of result lists (each with chunk_id, score, rank)
        k: RRF constant (typical: 60, higher = less aggressive)

    Returns:
        Merged and re-ranked results with rrf_score
    """
    rrf_scores = {}
    chunk_data = {}
    sources = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result["chunk_id"]

            # Initialize
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0
                chunk_data[chunk_id] = result
                sources[chunk_id] = []

            # Add RRF score contribution
            rrf_scores[chunk_id] += 1 / (k + rank)

            # Track source
            sources[chunk_id].append(result.get("source", "unknown"))

    # Sort by RRF score
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final result list
    merged = []
    for chunk_id, rrf_score in sorted_chunks:
        chunk = chunk_data[chunk_id].copy()
        chunk["rrf_score"] = rrf_score

        # Determine source tag
        chunk_sources = sources[chunk_id]
        if "elasticsearch" in chunk_sources and "milvus" in chunk_sources:
            chunk["source"] = "hybrid_overlap"
        elif "elasticsearch" in chunk_sources:
            chunk["source"] = "es_only"
        else:
            chunk["source"] = "milvus_only"

        merged.append(chunk)

    logger.info(f"RRF merge: {len(merged)} unique chunks from {len(result_lists)} sources")

    return merged


async def get_hybrid_context(
    query: str,
    keywords: str,
    es_client: Any,
    chunks_vdb: Any,
    text_chunks_storage: Any,
    embedding_manager: Any,
    llm_provider: Any,
    config: dict,
    query_param: Any,
) -> dict:
    """
    Retrieve chunks using hybrid search (Elasticsearch + Milvus).

    Combines:
    - Elasticsearch BM25 keyword search
    - Milvus semantic vector search with query expansion

    Merges with Reciprocal Rank Fusion (RRF).

    Args:
        query: Original user query
        keywords: Expanded keywords for ES search
        es_client: Elasticsearch client
        chunks_vdb: Milvus vector storage
        text_chunks_storage: Redis storage
        embedding_manager: Embedding manager
        llm_provider: LLM provider
        config: Global config dict
        query_param: QueryParam object

    Returns:
        {
            "chunks": [chunk objects],
            "sources": [source tags],
            "scores": {chunk_id: rrf_score},
            "es_count": int,
            "milvus_count": int,
            "overlap_count": int,
            "query_expansions": [str]
        }
    """
    from .vector_search import get_vector_context

    # Get hybrid search config
    hybrid_config = config.get("hybrid_search", {})
    es_top_k = hybrid_config.get("es_top_k", 50)
    milvus_top_k = hybrid_config.get("milvus_top_k", 50)
    merged_top_k = hybrid_config.get("merged_top_k", 100)
    rrf_k = hybrid_config.get("rrf_k", 60)
    fallback_on_error = hybrid_config.get("fallback_to_semantic_on_es_error", True)

    # 1. Elasticsearch BM25 search
    es_results = []
    try:
        es_results = await search_elasticsearch(keywords=keywords, es_client=es_client, top_k=es_top_k)
    except Exception as e:
        logger.error(f"Elasticsearch search failed: {e}")
        if not fallback_on_error:
            raise

    # 2. Milvus semantic search with query expansion (query variations for semantic diversity)
    milvus_results = []
    query_expansions = []
    try:
        vector_chunks, query_expansions = await get_vector_context(
            query=query,
            embedding_manager=embedding_manager,
            chunks_vdb=chunks_vdb,
            text_chunks_storage=text_chunks_storage,
            top_k=milvus_top_k,
            cosine_threshold=query_param.cosine_threshold,
            llm_provider=llm_provider,
            enable_query_expansion=config.get("ENABLE_CANDIDATE_FILTERING", True),
            num_query_expansions=config.get("NUM_QUERY_EXPANSIONS", 5),
            max_tokens_query_expansion=config.get("MAX_TOKENS_QUERY_EXPANSION", 5000),
            return_expansions=True,
        )

        # Convert to result format
        for rank, chunk in enumerate(vector_chunks, start=1):
            milvus_results.append(
                {
                    "chunk_id": chunk.get("id") or chunk.get("chunk_id"),  # Support both field names
                    "score": chunk.get("score", 0),
                    "content": chunk.get("content", ""),
                    "full_doc_id": chunk.get("full_doc_id", ""),
                    "chunk_order_index": chunk.get("chunk_order_index", 0),
                    "source": "milvus",
                    "rank": rank,
                }
            )
    except Exception as e:
        logger.error(f"Milvus search failed: {e}")
        if not fallback_on_error:
            raise

    # 3. Merge with RRF
    if es_results and milvus_results:
        merged_results = reciprocal_rank_fusion([es_results, milvus_results], k=rrf_k)
    elif es_results:
        logger.warning("Using ES-only results (Milvus failed or returned 0)")
        merged_results = es_results
    elif milvus_results:
        logger.warning("Using Milvus-only results (ES failed or returned 0)")
        merged_results = milvus_results
    else:
        logger.error("Both ES and Milvus returned 0 results!")
        merged_results = []

    # 4. Track statistics
    es_chunk_ids = {r["chunk_id"] for r in es_results}
    milvus_chunk_ids = {r["chunk_id"] for r in milvus_results}
    overlap_ids = es_chunk_ids & milvus_chunk_ids

    logger.info(
        f"Hybrid search: ES={len(es_chunk_ids)}, Milvus={len(milvus_chunk_ids)}, "
        f"Overlap={len(overlap_ids)}, Total unique={len(merged_results)}"
    )

    return {
        "chunks": merged_results[:merged_top_k],
        "sources": [c["source"] for c in merged_results[:merged_top_k]],
        "scores": {c["chunk_id"]: c.get("rrf_score", c.get("score", 0)) for c in merged_results[:merged_top_k]},
        "es_count": len(es_chunk_ids),
        "milvus_count": len(milvus_chunk_ids),
        "overlap_count": len(overlap_ids),
        "query_expansions": query_expansions,
    }
