"""
Chunk picking strategies for retrieval.

Adapted from plans/lightrag-code/retrieval/chunk_picking.py
"""

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..storage.base import BaseKVStorage, BaseVectorStorage

logger = logging.getLogger(__name__)

# Global cache for chunk similarity scores and ranks (for logging purposes)
# Format: chunk_id -> {"cosine_similarity": float, "similarity_rank": int}
_CHUNK_SIMILARITY_CACHE = {}


def store_similarity_metadata(chunk_id: str, similarity: float, rank: int):
    """Store similarity metadata for a chunk (for logging purposes)."""
    _CHUNK_SIMILARITY_CACHE[chunk_id] = {
        "cosine_similarity": similarity,
        "similarity_rank": rank,
    }


def get_similarity_metadata(chunk_id: str) -> dict | None:
    """Get stored similarity metadata for a chunk."""
    return _CHUNK_SIMILARITY_CACHE.get(chunk_id)


def clear_similarity_cache():
    """Clear the similarity cache (call at start of new query)."""
    global _CHUNK_SIMILARITY_CACHE
    _CHUNK_SIMILARITY_CACHE = {}


def pick_by_weighted_polling(
    entities_or_relations: list[dict],
    max_related_chunks: int,
    min_related_chunks: int = 1,
) -> list[str]:
    """
    Linear gradient weighted polling algorithm for text chunk selection.

    This algorithm ensures that entities/relations with higher importance get more text chunks,
    forming a linear decreasing allocation pattern.

    Args:
        entities_or_relations: List of entities or relations sorted by importance (high to low)
        max_related_chunks: Expected number of text chunks for the highest importance entity/relation
        min_related_chunks: Expected number of text chunks for the lowest importance entity/relation

    Returns:
        List of selected text chunk IDs
    """
    if not entities_or_relations:
        return []

    n = len(entities_or_relations)
    if n == 1:
        # Only one entity/relation, return its first max_related_chunks text chunks
        entity_chunks = entities_or_relations[0].get("sorted_chunks", [])
        return entity_chunks[:max_related_chunks]

    # Calculate expected text chunk count for each position (linear decrease)
    expected_counts = []
    for i in range(n):
        # Linear interpolation: from max_related_chunks to min_related_chunks
        ratio = i / (n - 1) if n > 1 else 0
        expected = max_related_chunks - ratio * (max_related_chunks - min_related_chunks)
        expected_counts.append(int(round(expected)))

    # First round allocation: allocate by expected values
    selected_chunks = []
    used_counts = []  # Track number of chunks used by each entity
    total_remaining = 0  # Accumulate remaining quotas

    for i, entity_rel in enumerate(entities_or_relations):
        entity_chunks = entity_rel.get("sorted_chunks", [])
        expected = expected_counts[i]

        # Actual allocatable count
        actual = min(expected, len(entity_chunks))
        selected_chunks.extend(entity_chunks[:actual])
        used_counts.append(actual)

        # Accumulate remaining quota
        remaining = expected - actual
        if remaining > 0:
            total_remaining += remaining

    # Second round allocation: multi-round scanning to allocate remaining quotas
    for _ in range(int(total_remaining)):
        allocated = False

        # Scan entities one by one, allocate one chunk when finding unused chunks
        for i, entity_rel in enumerate(entities_or_relations):
            entity_chunks = entity_rel.get("sorted_chunks", [])

            # Check if there are still unused chunks
            if used_counts[i] < len(entity_chunks):
                # Allocate one chunk
                selected_chunks.append(entity_chunks[used_counts[i]])
                used_counts[i] += 1
                allocated = True
                break

        # If no chunks were allocated in this round, all entities are exhausted
        if not allocated:
            break

    return selected_chunks


async def pick_by_entity_overlap(
    entities_or_relations: list[dict],
    text_chunks_storage: "BaseKVStorage",
    top_k: int = 10000,
    detailed_logger=None,
) -> list[str]:
    """
    Select chunks based on entity overlap count (chunk centrality).

    Chunks referenced by more entities are prioritized (hub chunks).
    Tie-breaking: chunks with the same entity count are sorted by content length (descending).

    Optimization: If total unique chunks < top_k, returns all chunks immediately without sorting.

    Args:
        entities_or_relations: List of entities or relations with "sorted_chunks" field
        text_chunks_storage: Storage to fetch chunk content for length-based tie-breaking
        top_k: Maximum number of chunks to return (default: 10000)

    Returns:
        List of selected chunk IDs sorted by entity count (desc), then content length (desc)
    """
    if not entities_or_relations:
        return []

    # Count how many entities reference each chunk
    chunk_entity_count = {}
    all_chunk_ids = set()

    for entity in entities_or_relations:
        entity_chunks = entity.get("sorted_chunks", [])
        for chunk_id in entity_chunks:
            all_chunk_ids.add(chunk_id)
            if chunk_id not in chunk_entity_count:
                chunk_entity_count[chunk_id] = 0
            chunk_entity_count[chunk_id] += 1

    total_chunks = len(all_chunk_ids)

    # Optimization: If total chunks < top_k, return all chunks immediately
    if total_chunks <= top_k:
        logger.info(
            f"Entity overlap selection: {total_chunks} total chunks <= {top_k} top_k, "
            f"returning all chunks (no sorting needed)"
        )
        return list(all_chunk_ids)

    logger.info(
        f"Entity overlap selection: {total_chunks} total chunks > {top_k} top_k, "
        f"ranking by entity count and content length"
    )

    # Fetch chunk content to get lengths for tie-breaking
    chunk_contents_list = await text_chunks_storage.get_by_ids(list(all_chunk_ids))

    # Build chunk_id -> content_length mapping
    chunk_lengths = {}
    for chunk_data in chunk_contents_list:
        chunk_id = chunk_data.get("id") or chunk_data.get("chunk_id")
        if chunk_id:
            content = chunk_data.get("content", "")
            chunk_lengths[chunk_id] = len(content)

    # Sort by: (1) entity count DESC, (2) content length DESC
    sorted_chunks = sorted(
        all_chunk_ids,
        key=lambda chunk_id: (
            -chunk_entity_count.get(chunk_id, 0),  # Primary: entity count (descending)
            -chunk_lengths.get(chunk_id, 0),  # Secondary: content length (descending)
        ),
    )

    # Detailed logging: Log ALL chunks BEFORE top-k selection
    if detailed_logger:
        for rank, chunk_id in enumerate(sorted_chunks, start=1):
            entity_count = chunk_entity_count.get(chunk_id, 0)
            detailed_logger.log_retrieval_kg_chunk(
                {
                    "chunk_id": chunk_id,
                    "entity_count": entity_count,
                    "rank": rank,
                    "selected": rank <= top_k,  # Whether it will be selected
                }
            )

    selected_chunks = sorted_chunks[:top_k]

    # Log statistics
    top_10 = selected_chunks[:10]
    logger.info("Entity overlap selection: Top 10 chunks by entity count:")
    for idx, chunk_id in enumerate(top_10, 1):
        entity_count = chunk_entity_count.get(chunk_id, 0)
        content_length = chunk_lengths.get(chunk_id, 0)
        logger.info(f"  {idx}. {chunk_id}: {entity_count} entities, {content_length} chars")

    logger.info(f"Selected {len(selected_chunks)} chunks using ENTITY_OVERLAP method (top_k={top_k})")

    return selected_chunks


async def get_candidate_chunks_from_vector_search(
    queries: list[str],
    chunks_vdb: "BaseVectorStorage",
    embedding_func: Callable,
    top_k_per_query: int = 500,
    final_top_k: int | None = None,
    detailed_logger=None,
) -> set[str]:
    """
    Get candidate chunks using vector search on multiple queries.

    This function performs pre-filtering to reduce the search space before
    fetching vectors for similarity computation.

    Args:
        queries: List of expanded queries from query expansion
        chunks_vdb: Vector database for chunks
        embedding_func: Embedding function (async)
        top_k_per_query: Top K chunks per query (default: 500)
        final_top_k: Final top K chunks after merging by max similarity (default: None = all)

    Returns:
        Set of candidate chunk IDs (union of all query results, limited to final_top_k by max similarity)
    """
    if not queries:
        return set()

    try:
        import time

        start_time = time.time()

        # Step 1: Compute embeddings for all queries (batch operation)
        logger.info(f"Computing embeddings for {len(queries)} expanded queries...")
        embed_start = time.time()
        query_embeddings = await embedding_func(queries)
        embed_time = time.time() - embed_start

        # Step 2: Perform vector search on chunks collection for each query IN PARALLEL
        logger.info(f"Searching chunks collection for {len(queries)} queries (top_k={top_k_per_query} per query)...")
        search_start = time.time()

        candidate_ids = set()

        # Define async search function for each query
        async def search_single_query(idx, query, embedding):
            try:
                # Query chunks collection using the MilvusStorage.query() method
                # MilvusStorage automatically adjusts ef parameter to satisfy: ef >= top_k
                results = await chunks_vdb.query(
                    query_text=query,
                    top_k=top_k_per_query,
                    query_embedding=embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
                    cosine_threshold=0.0,
                )

                logger.debug(f"Query {idx}/{len(queries)}: {len(results)} results")
                return results

            except Exception as e:
                logger.warning(f"Vector search failed for query {idx}: {e}")
                return []

        # Execute all searches in parallel
        import asyncio

        search_tasks = [
            search_single_query(idx, query, embedding)
            for idx, (query, embedding) in enumerate(zip(queries, query_embeddings), 1)
        ]
        all_results = await asyncio.gather(*search_tasks)

        # Collect all chunk IDs with max similarity score across queries
        # Also track which query each chunk came from and the rank within that query
        chunk_max_scores = {}  # chunk_id -> max_score
        chunk_query_info = {}  # chunk_id -> {"query_index": int, "rank": int, "score": float}

        for query_idx, results in enumerate(all_results):
            for rank, result in enumerate(results, 1):
                chunk_id = result.get("id")
                score = result.get("score", 0.0)
                if chunk_id:
                    # Track max score across all queries
                    if chunk_id not in chunk_max_scores or score > chunk_max_scores[chunk_id]:
                        chunk_max_scores[chunk_id] = score
                        chunk_query_info[chunk_id] = {"query_index": query_idx, "rank": rank, "score": score}

        # Log score distribution BEFORE limiting
        from ..query.kg.helpers import log_score_distribution

        chunks_with_scores = [{"id": cid, "score": score} for cid, score in chunk_max_scores.items()]
        log_score_distribution(chunks_with_scores, "score", "Milvus Vector Search (Before Limit)")

        # Sort by max score (descending) and limit to final_top_k
        if final_top_k and len(chunk_max_scores) > final_top_k:
            sorted_chunks = sorted(chunk_max_scores.items(), key=lambda x: x[1], reverse=True)
            candidate_ids = set(chunk_id for chunk_id, _ in sorted_chunks[:final_top_k])

            # Log score distribution AFTER limiting
            limited_chunks = [{"id": cid, "score": score} for cid, score in sorted_chunks[:final_top_k]]
            log_score_distribution(limited_chunks, "score", f"Milvus Vector Search (After top-{final_top_k} limit)")
        else:
            candidate_ids = set(chunk_max_scores.keys())
            sorted_chunks = sorted(chunk_max_scores.items(), key=lambda x: x[1], reverse=True)

        search_time = time.time() - search_start
        total_time = time.time() - start_time

        # Detailed logging for all chunks
        if detailed_logger:
            # Log each chunk to retrieval_milvus.jsonl
            final_chunks = sorted_chunks[:final_top_k] if final_top_k else sorted_chunks
            for global_rank, (chunk_id, score) in enumerate(final_chunks, 1):
                query_info = chunk_query_info.get(chunk_id, {})
                detailed_logger.log_retrieval_milvus_chunk(
                    {
                        "chunk_id": chunk_id,
                        "score": score,
                        "rank": global_rank,
                        "query_index": query_info.get("query_index", -1),
                        "query_rank": query_info.get("rank", -1),
                    }
                )

            # Prepare summary with score distribution
            scores_array = [score for _, score in final_chunks]
            if scores_array:
                summary_data = {
                    "total_chunks": len(candidate_ids),
                    "num_queries": len(queries),
                    "top_k_per_query": top_k_per_query,
                    "final_top_k": final_top_k,
                    "embed_time_ms": embed_time * 1000,
                    "search_time_ms": search_time * 1000,
                    "total_time_ms": total_time * 1000,
                    "score_stats": {
                        "mean": float(np.mean(scores_array)),
                        "std": float(np.std(scores_array)),
                        "min": float(np.min(scores_array)),
                        "max": float(np.max(scores_array)),
                        "median": float(np.median(scores_array)),
                    },
                }
                detailed_logger.log_retrieval_milvus_summary(summary_data)

        logger.info(
            f"Candidate generation: {len(chunk_max_scores)} unique chunks from {len(queries)} queries, "
            f"limited to {len(candidate_ids)} by final_top_k "
            f"in {total_time:.3f}s (embed: {embed_time:.3f}s, search: {search_time:.3f}s)"
        )

        return candidate_ids

    except Exception as e:
        logger.error(f"Candidate chunk generation failed: {e}")
        import traceback

        logger.error(f"Error: {traceback.format_exc()}")
        # Re-raise to fail the pipeline - candidate generation is critical for query quality
        raise


async def pick_by_vector_similarity(
    query: str,
    text_chunks_storage: "BaseKVStorage",
    chunks_vdb: "BaseVectorStorage",
    num_of_chunks: int,
    entity_info: list[dict[str, Any]],
    embedding_func: Callable,
    query_embedding=None,
    llm_provider=None,
    enable_candidate_filtering: bool = True,
    candidate_top_k: int = 1000,
    num_query_expansions: int = 2,
    min_query_expansions: int | None = None,
    max_query_expansions: int | None = None,
    min_intersection_size: int = 50,
    max_tokens_query_expansion: int = 1000,
    high_similarity_threshold: float = 0.95,
    query_expansions: list[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Vector similarity-based text chunk selection algorithm with query-guided pre-filtering.

    This algorithm uses LLM query expansion + vector search to pre-filter candidates
    before fetching all entity-related chunks, significantly reducing vector fetching overhead.

    Optimization strategy:
    1. Generate focused retrieval queries using LLM
    2. Vector search to get ~1000 candidate chunks
    3. Intersect candidates with entity-related chunks
    4. Only fetch vectors for filtered subset (~100-500 instead of 164K)

    Args:
        query: User's original query string
        text_chunks_storage: Text chunks storage instance
        chunks_vdb: Vector database storage for chunks
        num_of_chunks: Number of chunks to select
        entity_info: List of entity information containing chunk IDs
        embedding_func: Embedding function to compute query embedding
        query_embedding: Pre-computed query embedding (optional)
        llm_provider: LLM provider for query expansion (optional)
        enable_candidate_filtering: Enable query-guided pre-filtering (default: True)
        candidate_top_k: Candidate pool size for vector search (default: 1000)
        num_query_expansions: Number of query expansions (default: 2)
        min_intersection_size: Minimum intersection size to use filtering (default: 50)
        max_tokens_query_expansion: Max tokens for query expansion LLM (default: 1000)
        high_similarity_threshold: Chunks with similarity >= this threshold are always included,
            even if not in entity chunks (default: 0.95)

    Returns:
        Tuple of (selected_chunk_ids, failed_early_rerank_chunks)
        - selected_chunk_ids: List of selected text chunk IDs sorted by similarity (highest first)
        - failed_early_rerank_chunks: List of chunk dicts that failed early reranking threshold
    """
    import time

    # Clear similarity cache at start of new query to prevent memory leaks
    clear_similarity_cache()

    # Track failed early reranking chunks to pass to context builder
    failed_early_rerank_chunks = []

    entity_count = len(entity_info) if entity_info else 0
    logger.debug(f"Vector similarity chunk selection: num_of_chunks={num_of_chunks}, entity_info_count={entity_count}")

    if not entity_info or num_of_chunks <= 0:
        return [], []

    # Collect all unique chunk IDs from entity info
    all_entity_chunk_ids = set()
    for entity in entity_info:
        chunk_ids = entity.get("sorted_chunks", [])
        all_entity_chunk_ids.update(chunk_ids)

    if not all_entity_chunk_ids:
        logger.info(
            "Vector similarity chunk selection: no chunk IDs found in entity/relation info (expected for relationships)"
        )
        return [], []

    logger.info(f"Vector similarity: {len(all_entity_chunk_ids)} unique entity-related chunk IDs")

    # OPTIMIZATION: Query-guided pre-filtering to reduce vector fetching
    # Always enable query expansion when LLM provider is available
    if enable_candidate_filtering and (llm_provider or query_expansions):
        try:
            filter_start = time.time()

            # Step 1: Use provided query expansions or generate new ones
            if query_expansions:
                logger.info(f"Reusing {len(query_expansions)} query expansions for KG entity chunk filtering")
                expanded_queries = query_expansions
            else:
                # Generate expanded queries using LLM
                from .query_expansion import expand_query_for_retrieval

                expanded_queries = await expand_query_for_retrieval(
                    query=query,
                    llm_provider=llm_provider,
                    num_expansions=num_query_expansions,
                    min_expansions=min_query_expansions,
                    max_expansions=max_query_expansions,
                    enable=True,
                    max_tokens=max_tokens_query_expansion,
                    context="for KG entity chunk filtering",
                )

            # Step 2: Get candidate chunks via vector search
            candidate_ids = await get_candidate_chunks_from_vector_search(
                queries=expanded_queries,
                chunks_vdb=chunks_vdb,
                embedding_func=embedding_func,
                top_k_per_query=candidate_top_k // len(expanded_queries) if expanded_queries else candidate_top_k,
            )

            # Step 3: High-similarity bypass FIRST - always keep these chunks
            high_sim_chunks = set()
            if candidate_ids and high_similarity_threshold < 1.0:
                # Compute query embedding if not already computed
                if query_embedding is None:
                    query_embedding = await embedding_func([query])
                    query_embedding = query_embedding[0]

                # Get vectors for ALL candidates to find high-similarity ones
                from ..utils.text_processing import cosine_similarity

                logger.info(f"High-similarity bypass: Checking {len(candidate_ids)} candidates...")
                candidate_vectors = await chunks_vdb.get_vectors_by_ids(list(candidate_ids))

                # Calculate similarities for all candidates and sort them
                candidate_similarities = []
                for chunk_id in candidate_ids:
                    if chunk_id in candidate_vectors:
                        chunk_embedding = candidate_vectors[chunk_id]
                        try:
                            similarity = cosine_similarity(query_embedding, chunk_embedding)
                            candidate_similarities.append((chunk_id, similarity))
                            if similarity >= high_similarity_threshold:
                                high_sim_chunks.add(chunk_id)
                        except Exception as e:
                            logger.warning(f"Failed to calculate similarity for chunk {chunk_id}: {e}")

                # Sort by similarity (highest first) and log top 10
                candidate_similarities.sort(key=lambda x: x[1], reverse=True)
                top_10 = candidate_similarities[:10]

                # Fetch content for top 10 chunks to show snippets
                top_10_ids = [chunk_id for chunk_id, _ in top_10]
                top_10_contents_list = await text_chunks_storage.get_by_ids(top_10_ids)

                # Convert list to dict keyed by chunk_id
                top_10_contents = {}
                for chunk_data in top_10_contents_list:
                    chunk_id_key = chunk_data.get("id") or chunk_data.get("chunk_id")
                    if chunk_id_key:
                        top_10_contents[chunk_id_key] = chunk_data

                logger.info(f"Top 10 candidates by similarity (threshold={high_similarity_threshold:.2f}):")
                for idx, (chunk_id, similarity) in enumerate(top_10, 1):
                    chunk_content = top_10_contents.get(chunk_id, {})
                    content_text = chunk_content.get("content", "")
                    snippet = content_text[:150] + "..." if len(content_text) > 150 else content_text
                    snippet = snippet.replace("\n", " ")
                    logger.info(f"  {idx}. [{chunk_id}] sim={similarity:.4f} | {snippet}")

                if high_sim_chunks:
                    logger.info(
                        f"High-similarity bypass: Found {len(high_sim_chunks)} chunks "
                        f"with similarity >= {high_similarity_threshold:.2f} (will always keep these)"
                    )

            # Step 3.5: EARLY RERANKING - rerank high-sim chunks before fetching entity chunks
            # If enough chunks pass rerank threshold, skip entity chunk retrieval entirely
            if high_sim_chunks and llm_provider:
                try:
                    logger.info(
                        f"Early reranking: checking if {len(high_sim_chunks)} high-similarity chunks "
                        f"are sufficient (target={num_of_chunks})"
                    )

                    # Import reranking processor
                    from ..rerank.processor import RerankProcessor

                    # Get rerank config
                    rerank_config = {
                        "provider": "local",
                        "model": "jinaai/jina-reranker-v2-base-multilingual",
                        "device": "cuda:0",
                        "max_length": 1024,
                        "batch_size": 1000,
                        "min_score": 0.5,
                        "early_stop_target": 5,
                    }

                    # Try to get rerank processor from existing config
                    # This will be available when called from process_chunks_unified
                    rerank_processor = RerankProcessor(config=rerank_config)
                    await rerank_processor.initialize()

                    # Fetch content for high-sim chunks
                    high_sim_ids = list(high_sim_chunks)
                    high_sim_contents = await text_chunks_storage.get_by_ids(high_sim_ids)

                    # Convert to dict keyed by chunk_id
                    high_sim_dict = {}
                    for chunk_data in high_sim_contents:
                        chunk_id_key = chunk_data.get("id") or chunk_data.get("chunk_id")
                        if chunk_id_key:
                            high_sim_dict[chunk_id_key] = chunk_data

                    # Prepare documents for reranking
                    docs_for_rerank = []
                    for chunk_id in high_sim_ids:
                        if chunk_id in high_sim_dict:
                            content = high_sim_dict[chunk_id].get("content", "")
                            docs_for_rerank.append(
                                {
                                    "id": chunk_id,
                                    "text": content,
                                    "content": content,
                                }
                            )

                    if docs_for_rerank:
                        # Rerank high-sim chunks with multi-query if available
                        if len(expanded_queries) > 1:
                            reranked = await rerank_processor.rerank_multi_query(
                                queries=expanded_queries,
                                documents=docs_for_rerank,
                                top_k=None,
                                score_aggregation="max",
                            )
                        else:
                            reranked = await rerank_processor.rerank(
                                query=query,
                                documents=docs_for_rerank,
                                top_k=None,
                            )

                        # Count how many pass threshold
                        min_score = rerank_config.get("min_score", 0.5)
                        passed_chunks = [c for c in reranked if c.get("rerank_score", 0.0) >= min_score]
                        failed_chunks = [c for c in reranked if c.get("rerank_score", 0.0) < min_score]

                        logger.info(
                            f"Early reranking: {len(passed_chunks)}/{len(reranked)} high-similarity chunks "
                            f"passed threshold (min_score={min_score})"
                        )

                        # Update high_sim_chunks to only include chunks that passed reranking
                        # Failed chunks will be returned and passed to context builder
                        passed_chunk_ids = {c["id"] for c in passed_chunks}
                        original_high_sim_count = len(high_sim_chunks)
                        high_sim_chunks = high_sim_chunks.intersection(passed_chunk_ids)

                        if len(high_sim_chunks) < original_high_sim_count:
                            filtered_count = original_high_sim_count - len(high_sim_chunks)
                            logger.info(
                                f"Early reranking: {len(failed_chunks)} failed chunks saved for maybe-related section, "
                                f"filtered out {filtered_count} from final results"
                            )

                        # Store failed chunks to return them - they will be added to context as maybe_related_chunks
                        failed_early_rerank_chunks = failed_chunks

                        # If we have enough chunks, return early and skip entity chunk retrieval
                        if len(passed_chunks) >= num_of_chunks:
                            logger.info(
                                f"Early exit: {len(passed_chunks)} high-similarity chunks passed threshold "
                                f"(>= target {num_of_chunks}), skipping entity chunk retrieval"
                            )

                            # Sort by rerank score and return top num_of_chunks IDs
                            passed_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                            selected_chunk_ids = [c["id"] for c in passed_chunks[:num_of_chunks]]

                            return selected_chunk_ids, failed_early_rerank_chunks

                except Exception as e:
                    logger.warning(f"Early reranking failed, continuing with full pipeline: {e}")
                    # Continue with normal flow if early reranking fails

            # Step 4: Intersect remaining candidates with entity chunks
            if candidate_ids:
                # Remove high-similarity chunks from candidates before intersection
                remaining_candidates = candidate_ids - high_sim_chunks

                if remaining_candidates:
                    filtered_chunk_ids = all_entity_chunk_ids.intersection(remaining_candidates)

                    # Check intersection size
                    if len(filtered_chunk_ids) >= min_intersection_size:
                        filter_time = time.time() - filter_start
                        reduction = len(all_entity_chunk_ids) - len(filtered_chunk_ids)
                        reduction_pct = (reduction / len(all_entity_chunk_ids)) * 100 if all_entity_chunk_ids else 0

                        logger.info(
                            f"Pre-filtering: {len(filtered_chunk_ids)} chunks after intersection "
                            f"({reduction} chunks filtered, {reduction_pct:.1f}% reduction) in {filter_time:.3f}s"
                        )

                        # Combine high-similarity chunks + filtered entity chunks
                        all_chunk_ids = list(high_sim_chunks.union(filtered_chunk_ids))
                        logger.info(
                            f"Final selection: {len(high_sim_chunks)} high-similarity + "
                            f"{len(filtered_chunk_ids)} entity-intersected = {len(all_chunk_ids)} total"
                        )
                    else:
                        # Intersection too small, use high-similarity + all entity chunks
                        logger.warning(
                            f"Pre-filtering: intersection too small "
                            f"({len(filtered_chunk_ids)} < {min_intersection_size}), "
                            f"using {len(high_sim_chunks)} high-similarity + all entity chunks"
                        )
                        all_chunk_ids = list(high_sim_chunks.union(all_entity_chunk_ids))
                else:
                    # All candidates were high-similarity, just use them + entity chunks
                    logger.info(
                        f"All candidates were high-similarity, "
                        f"using {len(high_sim_chunks)} high-sim chunks + entity chunks"
                    )
                    all_chunk_ids = list(high_sim_chunks.union(all_entity_chunk_ids))
            else:
                # No candidates, fallback to entity chunks + any high-sim we found
                logger.warning("Pre-filtering: no candidates found, falling back to entity chunks")
                combined = high_sim_chunks.union(all_entity_chunk_ids) if high_sim_chunks else all_entity_chunk_ids
                all_chunk_ids = list(combined)

        except Exception as e:
            logger.error(f"Pre-filtering failed: {e}")
            # Re-raise to fail the entire pipeline - query expansion failure means low quality results
            raise
    else:
        # Pre-filtering disabled or not applicable
        if not enable_candidate_filtering:
            logger.debug("Pre-filtering disabled by configuration")
        elif not llm_provider:
            logger.debug("Pre-filtering skipped: no LLM provider")
        else:
            logger.debug(f"Pre-filtering skipped: chunk count ({len(all_entity_chunk_ids)}) below threshold")

        all_chunk_ids = list(all_entity_chunk_ids)

    try:
        # Import cosine_similarity from text_processing
        from ..utils.text_processing import cosine_similarity

        # Use pre-computed query embedding if provided, otherwise compute it
        if query_embedding is None:
            query_embedding = await embedding_func([query])
            query_embedding = query_embedding[0]  # Extract first embedding from batch result
            logger.debug("Computed query embedding for vector similarity chunk selection")
        else:
            logger.debug("Using pre-computed query embedding for vector similarity chunk selection")

        # Get chunk embeddings from vector database
        chunk_vectors = await chunks_vdb.get_vectors_by_ids(all_chunk_ids)
        logger.debug(f"Vector similarity chunk selection: {len(chunk_vectors)} chunk vectors retrieved")

        if not chunk_vectors:
            logger.warning("Vector similarity chunk selection: no vectors retrieved from chunks_vdb")
            return [], failed_early_rerank_chunks

        # Log if some vectors are missing (but continue with what we have)
        if len(chunk_vectors) != len(all_chunk_ids):
            missing_count = len(all_chunk_ids) - len(chunk_vectors)
            logger.info(
                f"Vector similarity: {missing_count}/{len(all_chunk_ids)} chunk vectors missing, "
                f"continuing with {len(chunk_vectors)} vectors"
            )

        # Calculate cosine similarities
        similarities = []
        valid_vectors = 0
        for chunk_id in all_chunk_ids:
            if chunk_id in chunk_vectors:
                chunk_embedding = chunk_vectors[chunk_id]
                try:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append((chunk_id, similarity))
                    valid_vectors += 1
                except Exception as e:
                    logger.warning(
                        f"Vector similarity chunk selection: failed to calculate similarity for chunk {chunk_id}: {e}"
                    )
            else:
                logger.warning(f"Vector similarity chunk selection: no vector found for chunk {chunk_id}")

        # Sort by similarity (highest first) and select top num_of_chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        selected_chunks = [chunk_id for chunk_id, _ in similarities[:num_of_chunks]]

        # Store similarity metadata for all chunks (for logging during reranking)
        for rank, (chunk_id, similarity) in enumerate(similarities, 1):
            store_similarity_metadata(chunk_id, similarity, rank)

        logger.info(
            f"Vector similarity chunk selection: {len(selected_chunks)} chunks selected "
            f"from {len(all_chunk_ids)} candidates"
        )

        return selected_chunks, failed_early_rerank_chunks

    except Exception as e:
        logger.error(f"[VECTOR_SIMILARITY] Error in vector similarity sorting: {e}")
        import traceback

        logger.error(f"[VECTOR_SIMILARITY] Traceback: {traceback.format_exc()}")
        # Re-raise to fail the pipeline - vector similarity is critical for ranking quality
        raise


async def process_chunks_unified(
    query: str,
    unique_chunks: list[dict],
    query_param: Any,  # QueryParam object
    global_config: dict,
    source_type: str = "mixed",
    chunk_token_limit: int | None = None,
    chunks_vdb: "BaseVectorStorage | None" = None,
    query_expansions: list[str] | None = None,
    detailed_logger=None,
) -> list[dict]:
    """
    Unified processing for text chunks: deduplication, chunk_top_k limiting, reranking, and token truncation.

    NEW: If query_expansions is provided, uses multi-query reranking.

    Args:
        query: Search query for reranking
        unique_chunks: List of text chunks to process
        query_param: Query parameters containing configuration
        global_config: Global configuration dictionary
        source_type: Source type for logging ("vector", "entity", "relationship", "mixed")
        chunk_token_limit: Dynamic token limit for chunks (if None, uses default)
        chunks_vdb: Vector storage for chunks (needed for document expansion)
        query_expansions: List of expanded queries for multi-query reranking (optional)

    Returns:
        Processed and filtered list of text chunks with original chunk IDs preserved
    """
    from ..rerank.processor import apply_rerank_if_enabled
    from ..utils.text_processing import truncate_list_by_token_size

    if not unique_chunks:
        return []

    origin_count = len(unique_chunks)

    # 1. Apply reranking if enabled and query is provided
    # Skip if chunks already have rerank_score (to avoid double reranking)
    already_reranked = all(chunk.get("rerank_score") is not None for chunk in unique_chunks)

    if query_param.enable_rerank and query and unique_chunks and not already_reranked:
        rerank_top_k = query_param.chunk_top_k or len(unique_chunks)
        unique_chunks = await apply_rerank_if_enabled(
            query=query,
            retrieved_docs=unique_chunks,
            global_config=global_config,
            enable_rerank=query_param.enable_rerank,
            top_n=rerank_top_k,
            chunks_vdb=chunks_vdb,
            query_expansions=query_expansions,
            detailed_logger=detailed_logger,
        )
    elif already_reranked:
        logger.debug(f"Skipping reranking: {len(unique_chunks)} chunks already have rerank_score")

    # Note: Rerank score filtering is already handled inside apply_rerank_if_enabled()
    # No need for duplicate filtering here

    # 2. Apply chunk_top_k limiting if specified
    if query_param.chunk_top_k is not None and query_param.chunk_top_k > 0:
        if len(unique_chunks) > query_param.chunk_top_k:
            unique_chunks = unique_chunks[: query_param.chunk_top_k]
        logger.debug(f"Kept chunk_top-k: {len(unique_chunks)} chunks (deduplicated original: {origin_count})")

    # 3. Token-based final truncation
    tokenizer = global_config.get("tokenizer")
    if tokenizer and unique_chunks:
        # Set default chunk_token_limit if not provided
        if chunk_token_limit is None:
            # Get default from query_param or global_config
            chunk_token_limit = getattr(
                query_param,
                "max_total_tokens",
                global_config.get("MAX_TOTAL_TOKENS", 32000),
            )

        original_count = len(unique_chunks)

        unique_chunks = truncate_list_by_token_size(
            unique_chunks,
            key=lambda x: "\n".join(json.dumps(item, ensure_ascii=False) for item in [x]),
            max_token_size=chunk_token_limit,
            tokenizer=tokenizer,
        )

        logger.debug(
            f"Token truncation: {len(unique_chunks)} chunks from {original_count} "
            f"(chunk available tokens: {chunk_token_limit}, source: {source_type})"
        )

    # 4. Ensure id field is present (keep original chunk ID, don't replace with DC notation)
    final_chunks = []
    for i, chunk in enumerate(unique_chunks):
        chunk_with_id = chunk.copy()
        # Preserve original chunk ID if present, otherwise use fallback
        if "id" not in chunk_with_id or not chunk_with_id["id"]:
            chunk_with_id["id"] = chunk_with_id.get("chunk_id", f"chunk-unknown-{i + 1}")
        final_chunks.append(chunk_with_id)

    return final_chunks
