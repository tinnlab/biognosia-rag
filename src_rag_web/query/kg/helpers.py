"""
Common helper functions for KG query modes.

Extracts repetitive code blocks used across all KG modes.
"""

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def log_score_distribution(chunks: list[dict], score_key: str, stage_name: str):
    """
    Log detailed score distribution statistics for a stage.

    Args:
        chunks: List of chunks with scores
        score_key: Key to extract score from chunk dict (e.g., "score", "rerank_score")
        stage_name: Name of the stage for logging
    """
    if not chunks:
        logger.info(f"[{stage_name}] No chunks to analyze")
        return

    # Extract scores
    scores = []
    for chunk in chunks:
        score = chunk.get(score_key)
        if score is not None:
            scores.append(float(score))

    if not scores:
        logger.info(f"[{stage_name}] No scores found (key='{score_key}')")
        return

    scores = np.array(scores)

    # Basic statistics
    mean_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    std_score = np.std(scores)
    median_score = np.median(scores)

    logger.info(f"=== [{stage_name}] SCORE DISTRIBUTION ===")
    logger.info(f"  Total chunks: {len(scores)}")
    logger.info(f"  Mean: {mean_score:.4f} | Std: {std_score:.4f}")
    logger.info(f"  Min: {min_score:.4f} | Median: {median_score:.4f} | Max: {max_score:.4f}")

    # Percentile analysis
    percentiles = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95]
    logger.info("  Percentile distribution:")
    for p in percentiles:
        # Top p% means (100-p) percentile
        threshold = np.percentile(scores, 100 - p)
        count_above = np.sum(scores >= threshold)
        logger.info(f"    Top {p:2d}% ({count_above:5d} chunks): score >= {threshold:.4f}")

    # Log top 10 chunks with content preview
    logger.info(f"\n  === Top 10 Chunks for {stage_name} ===")
    sorted_chunks = sorted(chunks, key=lambda x: x.get(score_key, 0), reverse=True)[:10]
    for i, chunk in enumerate(sorted_chunks, 1):
        score = chunk.get(score_key, 0)
        chunk_id = chunk.get("id") or chunk.get("chunk_id", "unknown")
        content = chunk.get("content", "")
        content_preview = content[:150] if content else "(no content)"
        logger.info(f"    [{i}] Score: {score:.4f} | ID: {chunk_id}")
        logger.info(f"        Content: {content_preview}...")

    logger.info(f"=== END [{stage_name}] ===\n")


async def extract_entities_ngram(
    query: str, config: dict, storage_dict: dict, embedding_manager, detailed_logger=None
) -> tuple[list, bool]:
    """
    Extract entities using n-gram matching (Stage 1 + optional Stage 2).

    Returns:
        Tuple of (entity_info list, success boolean)
    """
    ngram_config = config.get("ngram_entity_matching", {})
    use_ngram = ngram_config.get("enable", False)

    if not use_ngram:
        logger.debug("N-gram matching disabled in config")
        return [], False

    logger.info("Starting n-gram entity extraction")
    start_time = time.time()

    try:
        from ...entity_extraction import NGramEntityMatcher

        entity_stats_manager = config.get("entity_stats_manager")

        # Stage 2 configuration
        stage2_config_section = config.get("second_stage_entity_discovery", {})
        enable_stage2 = stage2_config_section.get("enable", False)

        # Build Stage 2 config dict
        stage2_config = {
            "num_shuffles": stage2_config_section.get("num_shuffles", 1000),
            "appearance_threshold": stage2_config_section.get("appearance_threshold", 0.5),
            "min_similarity": stage2_config_section.get("min_similarity", 0.85),
            "top_k": stage2_config_section.get("top_k", 10),
            "min_entities": stage2_config_section.get("min_entities", 2),
            "embedding_batch_size": stage2_config_section.get("embedding_batch_size", 200),
            "query_batch_size": stage2_config_section.get("query_batch_size", 100),
            "shuffle_strategy": stage2_config_section.get("shuffle_strategy", "adaptive"),
            "max_exhaustive_snippets": stage2_config_section.get("max_exhaustive_snippets", 6),
            "min_unique_ratio": stage2_config_section.get("min_unique_ratio", 0.8),
            "exclude_collections": stage2_config_section.get("exclude_collections", []),
            # Add reranking support for Stage 2
            "rerank_processor": config.get("rerank_processor"),
            "rerank_config": config.get("rerank_entity", {}),
        }

        # Log n-gram config
        logger.debug(
            f"N-gram config: k_values={ngram_config.get('k_values')}, "
            f"min_similarity={ngram_config.get('min_similarity', 0.85)}, "
            f"max_pct_paper={ngram_config.get('max_pct_paper')}, "
            f"max_pct_chunk={ngram_config.get('max_pct_chunk')}, "
            f"max_num_text_variations={ngram_config.get('max_num_text_variations')}"
        )

        if enable_stage2:
            logger.debug(
                f"Stage 2 config: num_shuffles={stage2_config['num_shuffles']}, "
                f"appearance_threshold={stage2_config['appearance_threshold']}, "
                f"strategy={stage2_config['shuffle_strategy']}"
            )

        ngram_matcher = NGramEntityMatcher(
            embedding_manager=embedding_manager,
            milvus_storage=storage_dict.get("entities_vdb"),
            k_values=ngram_config.get("k_values", [1, 2, 3, 4, 5]),
            entity_stats_manager=entity_stats_manager,
            max_pct_paper=ngram_config.get("max_pct_paper"),
            max_pct_chunk=ngram_config.get("max_pct_chunk"),
            max_num_text_variations=ngram_config.get("max_num_text_variations"),
            min_similarity=ngram_config.get("min_similarity", 0.85),
            entity_collections=ngram_config.get("entity_collections"),
            enable_cache=ngram_config.get("enable_cache", True),
            max_cache_size=ngram_config.get("max_cache_size", 10000),
            enable_stage2=enable_stage2,
            stage2_config=stage2_config,
        )

        # New return format: (entities, metadata)
        entity_info, metadata = await ngram_matcher.match_query_entities(query, detailed_logger)

        # Normalize field names: entity_name -> entity_id
        for entity in entity_info:
            if "entity_name" in entity:
                entity["entity_id"] = entity["entity_name"]

        elapsed = time.time() - start_time
        logger.info(
            f"N-gram matching: extracted {len(entity_info)} entities in {elapsed:.3f}s "
            f"(Stage 1: {metadata['stage1_count']}, Stage 2: {metadata['stage2_count']})"
        )

        # Log top entities at debug level
        if entity_info:
            top_entities = entity_info[:5]
            logger.debug(f"Top entities: {[e.get('entity_id', 'unknown') for e in top_entities]}")

        return entity_info, metadata  # Return metadata instead of True

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"N-gram matching failed after {elapsed:.3f}s: {e}", exc_info=True)
        return [], False


async def get_entities_from_vector_search(
    query: str, embedding_manager, entities_vdb, graph_storage, top_k: int, cosine_threshold: float
) -> list:
    """
    Get entities via vector search + Neo4j lookup.

    Returns:
        List of entity info dicts with hashed_id added
    """
    from ...retrieval.kg_search import get_node_data
    from ...retrieval.vector_search import get_entity_vector_context

    logger.info(f"Entity vector search (fallback): top_k={top_k}, threshold={cosine_threshold}")
    start_time = time.time()

    # Vector search
    vector_start = time.time()
    entity_search_results = await get_entity_vector_context(
        query=query,
        embedding_manager=embedding_manager,
        entities_vdb=entities_vdb,
        top_k=top_k * 2,
        cosine_threshold=cosine_threshold,
    )
    vector_elapsed = time.time() - vector_start
    logger.debug(f"Entity vector search: {len(entity_search_results)} results in {vector_elapsed:.3f}s")

    # Build ID mapping
    entity_id_mapping = {}
    for result in entity_search_results:
        semantic_id = result.get("entity_name")
        hashed_id = result.get("id")
        if semantic_id and hashed_id:
            entity_id_mapping[semantic_id] = hashed_id

    # Entity metadata lookup (tries Milvus first, falls back to Neo4j)
    metadata_start = time.time()
    entity_ids = [result.get("entity_name") or result.get("id") for result in entity_search_results]
    entity_info = await get_node_data(
        node_ids=entity_ids,
        graph_storage=graph_storage,
        entities_vdb=entities_vdb,  # Will query Milvus for clean descriptions
        top_k=top_k,
    )
    metadata_elapsed = time.time() - metadata_start

    # Add hashed_id to entity_info
    for entity in entity_info:
        semantic_id = entity.get("entity_id")
        if semantic_id in entity_id_mapping:
            entity["hashed_id"] = entity_id_mapping[semantic_id]

    elapsed = time.time() - start_time
    logger.info(
        f"Entity retrieval: {len(entity_info)} entities in {elapsed:.3f}s "
        f"(vector: {vector_elapsed:.3f}s, metadata: {metadata_elapsed:.3f}s)"
    )

    # Log top entities
    if entity_info:
        top_entities = entity_info[:5]
        logger.debug(f"Top entities: {[e.get('entity_id', 'unknown') for e in top_entities]}")

    return entity_info


async def apply_entity_reranking(
    query: str,
    entity_info: list,
    config: dict,
) -> list:
    """
    Apply entity reranking to filter entities by relevance.

    This reduces the number of entities before chunk retrieval,
    which significantly improves performance.

    Args:
        query: User's query
        entity_info: List of entity dicts
        config: Global configuration dict

    Returns:
        Filtered list of entities that pass relevance threshold
    """
    from ...rerank.processor import rerank_entities

    # Check if empty
    if not entity_info:
        logger.debug("Entity reranking: no entities to rerank")
        return entity_info

    rerank_entity_config = config.get("rerank_entity", {})
    logger.info(
        f"Entity reranking config: enable_stage1={rerank_entity_config.get('enable_stage1', False)}, "
        f"enable_stage2={rerank_entity_config.get('enable_stage2', False)}, "
        f"min_score={rerank_entity_config.get('min_score', 0.3)}"
    )

    # Check if any reranking is enabled
    if not rerank_entity_config.get("enable_stage1", False) and not rerank_entity_config.get("enable_stage2", False):
        logger.info("Entity reranking: DISABLED for both stages in config, skipping reranking")
        return entity_info

    rerank_processor = config.get("rerank_processor")
    if not rerank_processor:
        logger.warning("Entity reranking: ENABLED but no rerank processor available, skipping")
        return entity_info

    logger.info(f"Entity reranking: STARTING for {len(entity_info)} entities")

    # Separate entities by stage (check for stage2_appearance_count field)
    stage1_entities = [e for e in entity_info if "stage2_appearance_count" not in e]
    stage2_entities = [e for e in entity_info if "stage2_appearance_count" in e]

    logger.info(f"Entity split: {len(stage1_entities)} Stage 1, {len(stage2_entities)} Stage 2")

    filtered_entities = []

    # Rerank Stage 1 entities if enabled
    if stage1_entities and rerank_entity_config.get("enable_stage1", False):
        logger.info(f"Reranking {len(stage1_entities)} Stage 1 entities...")
        stage1_filtered = await rerank_entities(
            query=query,
            entities=stage1_entities,
            rerank_config=rerank_entity_config,
            rerank_processor=rerank_processor,
            stage="stage1",
        )
        filtered_entities.extend(stage1_filtered)
        logger.info(f"Stage 1 reranking: kept {len(stage1_filtered)}/{len(stage1_entities)} entities")
    elif stage1_entities:
        # Stage 1 reranking disabled, keep all
        logger.info(f"Stage 1 reranking disabled, keeping all {len(stage1_entities)} Stage 1 entities")
        filtered_entities.extend(stage1_entities)

    # Stage 2 entities - check if already reranked (scores added in semantic_community.py)
    if stage2_entities:
        # Check if entities already have rerank scores (from semantic_community.py)
        already_reranked = any("entity_rerank_score" in e for e in stage2_entities)

        if already_reranked:
            # Already reranked in semantic_community.py, just filter by threshold
            logger.info("Stage 2 entities already reranked in discovery phase, applying threshold filter")
            min_score = rerank_entity_config.get("min_score", 0.3)
            stage2_filtered = [e for e in stage2_entities if e.get("entity_rerank_score", 0.0) >= min_score]
            filtered_entities.extend(stage2_filtered)
            logger.info(
                f"Stage 2 filtering: kept {len(stage2_filtered)}/{len(stage2_entities)} entities "
                f"(min_score={min_score})"
            )
        elif rerank_entity_config.get("enable_stage2", False):
            # Not reranked yet, apply reranking now
            logger.info(f"Reranking {len(stage2_entities)} Stage 2 entities...")
            stage2_filtered = await rerank_entities(
                query=query,
                entities=stage2_entities,
                rerank_config=rerank_entity_config,
                rerank_processor=rerank_processor,
                stage="stage2",
            )
            filtered_entities.extend(stage2_filtered)
            logger.info(f"Stage 2 reranking: kept {len(stage2_filtered)}/{len(stage2_entities)} entities")
        else:
            # Stage 2 reranking disabled, keep all
            logger.info(f"Stage 2 reranking disabled, keeping all {len(stage2_entities)} Stage 2 entities")
            filtered_entities.extend(stage2_entities)

    logger.info(f"Entity reranking: COMPLETED, kept {len(filtered_entities)}/{len(entity_info)} entities total")

    return filtered_entities


async def get_chunks_from_entities(
    entity_info: list,
    text_chunks_storage,
    chunk_entity_relation_storage,
    chunks_vdb,
    query: str,
    embedding_func,
    param,
    llm_provider=None,
    query_expansions: list[str] | None = None,
) -> tuple[list, list]:
    """
    Get chunks related to entities.

    Returns:
        Tuple of (chunks, failed_early_rerank_chunks)
        - chunks: List of chunk dicts with id and content
        - failed_early_rerank_chunks: List of chunk dicts that failed early reranking threshold
    """
    from ...retrieval.kg_search import find_related_text_unit_from_entities

    if not entity_info:
        logger.debug("No entities, skipping chunk retrieval")
        return [], []

    logger.info(
        f"Retrieving chunks for {len(entity_info)} entities "
        f"(method: {param.kg_chunk_pick_method}, kg_chunk_top_k={param.kg_chunk_top_k})"
    )
    start_time = time.time()

    entity_chunk_ids, failed_early_rerank_chunks = await find_related_text_unit_from_entities(
        entity_info=entity_info,
        text_chunks_storage=text_chunks_storage,
        chunk_entity_relation_storage=chunk_entity_relation_storage,
        chunks_vdb=chunks_vdb,
        num_of_chunks=param.kg_chunk_top_k,
        kg_chunk_pick_method=param.kg_chunk_pick_method,
        max_related_chunks=param.max_related_chunks,
        min_related_chunks=param.min_related_chunks,
        query=query,
        embedding_func=embedding_func,
        llm_provider=llm_provider,
        enable_candidate_filtering=param.enable_candidate_filtering,
        candidate_top_k=param.candidate_top_k,
        num_query_expansions=param.num_query_expansions,
        min_query_expansions=param.min_query_expansions,
        max_query_expansions=param.max_query_expansions,
        min_intersection_size=param.min_intersection_size,
        max_tokens_query_expansion=param.max_tokens_query_expansion,
        high_similarity_threshold=param.high_similarity_threshold,
        query_expansions=query_expansions if param.enable_vector_retrieval else None,
    )

    # Return chunk IDs only (content will be fetched later in batch)
    chunks = [{"id": cid, "source": "entity"} for cid in entity_chunk_ids]

    elapsed = time.time() - start_time
    logger.info(f"Entity chunks: retrieved {len(chunks)} chunk IDs in {elapsed:.3f}s")
    logger.debug(f"Chunk IDs (first 5): {entity_chunk_ids[:5]}")

    return chunks, failed_early_rerank_chunks


async def get_chunks_from_relations(
    relationship_info: list,
    text_chunks_storage,
    chunk_entity_relation_storage,
    chunks_vdb,
    query: str,
    embedding_func,
    param,
    llm_provider=None,
    query_expansions: list[str] | None = None,
) -> tuple[list, list]:
    """
    Get chunks related to relationships.

    Returns:
        Tuple of (chunks, failed_early_rerank_chunks)
        - chunks: List of chunk dicts with id and content
        - failed_early_rerank_chunks: List of chunk dicts that failed early reranking threshold
    """
    from ...retrieval.kg_search import find_related_text_unit_from_relations

    if not relationship_info:
        logger.debug("No relationships, skipping chunk retrieval")
        return [], []

    logger.info(
        f"Retrieving chunks for {len(relationship_info)} relationships "
        f"(method: {param.kg_chunk_pick_method}, kg_chunk_top_k={param.kg_chunk_top_k})"
    )
    start_time = time.time()

    relation_chunk_ids, failed_early_rerank_chunks = await find_related_text_unit_from_relations(
        relation_info=relationship_info,
        text_chunks_storage=text_chunks_storage,
        chunk_entity_relation_storage=chunk_entity_relation_storage,
        chunks_vdb=chunks_vdb,
        num_of_chunks=param.kg_chunk_top_k,
        kg_chunk_pick_method=param.kg_chunk_pick_method,
        max_related_chunks=param.max_related_chunks,
        min_related_chunks=param.min_related_chunks,
        query=query,
        embedding_func=embedding_func,
        llm_provider=llm_provider,
        enable_candidate_filtering=param.enable_candidate_filtering,
        candidate_top_k=param.candidate_top_k,
        num_query_expansions=param.num_query_expansions,
        min_query_expansions=param.min_query_expansions,
        max_query_expansions=param.max_query_expansions,
        min_intersection_size=param.min_intersection_size,
        max_tokens_query_expansion=param.max_tokens_query_expansion,
        high_similarity_threshold=param.high_similarity_threshold,
        query_expansions=query_expansions if param.enable_vector_retrieval else None,
    )

    # Return chunk IDs only (content will be fetched later in batch)
    chunks = [{"id": cid, "source": "relationship"} for cid in relation_chunk_ids]

    elapsed = time.time() - start_time
    logger.info(f"Relationship chunks: retrieved {len(chunks)} chunk IDs in {elapsed:.3f}s")
    logger.debug(f"Chunk IDs (first 5): {relation_chunk_ids[:5]}")

    return chunks, failed_early_rerank_chunks


async def process_and_respond(
    query: str,
    entity_info: list,
    relationship_info: list,
    vector_chunks: list,
    entity_chunks: list,
    relation_chunks: list,
    llm_provider,
    config: dict,
    param,
    mode: str,
    chunks_vdb=None,
    query_expansions: list[str] | None = None,
    failed_early_rerank_chunks: list[dict] | None = None,
    text_chunks_storage=None,
    detailed_logger=None,
    entities_only: bool = False,
) -> dict[str, Any]:
    """
    Common final processing: merge chunks, rerank, build context, generate LLM response.

    NEW: Passes query_expansions to chunk processing for multi-query reranking.
    NEW: Accepts failed_early_rerank_chunks from early reranking failures in vector search.

    Args:
        query_expansions: List of expanded queries for multi-query reranking (optional)
        failed_early_rerank_chunks: Chunks that failed early reranking threshold (optional)

    Returns:
        Query result dictionary
    """
    from ...llm import generate_llm_response
    from ...retrieval.chunk_picking import process_chunks_unified
    from ...retrieval.context_builder import (
        build_llm_context,
        build_query_context,
        format_references,
        merge_all_chunks,
    )

    logger.info(
        f"Final processing: {len(vector_chunks)} vector + {len(entity_chunks)} entity + "
        f"{len(relation_chunks)} relation chunks"
    )

    # Merge all chunks
    merge_start = time.time()
    merged_chunks = merge_all_chunks(
        vector_chunks=vector_chunks,
        entity_chunks=entity_chunks,
        relationship_chunks=relation_chunks,
    )
    merge_elapsed = time.time() - merge_start
    logger.info(f"Merged: {len(merged_chunks)} unique chunks in {merge_elapsed:.3f}s")

    # Apply merged_top_k limit with rank-based source balancing
    hybrid_config = config.get("hybrid_search", {})
    merged_top_k = hybrid_config.get("merged_top_k")

    if merged_top_k and len(merged_chunks) > merged_top_k:
        logger.info(f"Applying merged_top_k limit with rank-based balancing: {len(merged_chunks)} → {merged_top_k}")

        # Combine entity + relation into "kg" source for 3-source model
        for chunk in merged_chunks:
            sources = chunk.get("sources", [])
            if "entity" in sources or "relation" in sources:
                # Replace entity/relation with kg
                new_sources = set(sources) - {"entity", "relation"}
                new_sources.add("kg")
                chunk["sources"] = list(new_sources)

        # Rank chunks within each source by their original scores
        # We'll use the position in the original list as implicit ranking
        # (assuming chunks from each source are already sorted by score)
        source_ranks = {}  # chunk_id -> {source -> rank}

        for chunk in merged_chunks:
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            if chunk_id not in source_ranks:
                source_ranks[chunk_id] = {}

        # Assign ranks within each source (lower rank = better)
        kg_chunks = [c for c in merged_chunks if "kg" in c.get("sources", [])]
        es_chunks = [c for c in merged_chunks if "elasticsearch" in c.get("sources", [])]
        milvus_chunks = [c for c in merged_chunks if "milvus" in c.get("sources", [])]

        for rank, chunk in enumerate(kg_chunks, 1):
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            source_ranks[chunk_id]["kg"] = rank

        for rank, chunk in enumerate(es_chunks, 1):
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            source_ranks[chunk_id]["elasticsearch"] = rank

        for rank, chunk in enumerate(milvus_chunks, 1):
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            source_ranks[chunk_id]["milvus"] = rank

        # Compute average rank for each chunk
        for chunk in merged_chunks:
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            ranks = source_ranks.get(chunk_id, {})
            if ranks:
                chunk["avg_rank"] = sum(ranks.values()) / len(ranks)
                chunk["source_count"] = len(chunk.get("sources", []))
            else:
                chunk["avg_rank"] = float("inf")
                chunk["source_count"] = 0

        # Bucket by source count (descending: 3, 2, 1)
        buckets = {}
        for chunk in merged_chunks:
            count = chunk["source_count"]
            if count not in buckets:
                buckets[count] = []
            buckets[count].append(chunk)

        # Sort buckets by average rank within each bucket
        for count in buckets:
            buckets[count].sort(key=lambda x: x["avg_rank"])

        logger.info("  Source count distribution:")
        for count in sorted(buckets.keys(), reverse=True):
            logger.info(f"    {count} sources: {len(buckets[count])} chunks")

        # Fill strategy: always keep all highest-count bucket first
        selected = []
        remaining_budget = merged_top_k

        for source_count in sorted(buckets.keys(), reverse=True):
            bucket_chunks = buckets[source_count]

            if source_count == max(buckets.keys()):
                # Always keep all chunks with max source count (high confidence)
                selected.extend(bucket_chunks)
                logger.info(f"  Keeping all {len(bucket_chunks)} chunks with {source_count} sources (high confidence)")
                remaining_budget -= len(bucket_chunks)
            elif remaining_budget > 0:
                # Distribute remaining budget
                if len(bucket_chunks) <= remaining_budget:
                    # Can fit all chunks from this bucket
                    selected.extend(bucket_chunks)
                    logger.info(f"  Keeping all {len(bucket_chunks)} chunks with {source_count} sources")
                    remaining_budget -= len(bucket_chunks)
                else:
                    # Need to sample from this bucket with source balancing
                    logger.info(
                        f"  Sampling {remaining_budget} from {len(bucket_chunks)} chunks with {source_count} sources"
                    )

                    # Group chunks by individual source (chunks can belong to multiple groups)
                    source_names = ["kg", "elasticsearch", "milvus"]
                    by_source = {source: [] for source in source_names}

                    for chunk in bucket_chunks:
                        for source in chunk.get("sources", []):
                            if source in by_source:
                                by_source[source].append(chunk)

                    # Equal allocation per source
                    per_source_allocation = remaining_budget // len(source_names)
                    unallocated = remaining_budget % len(source_names)

                    logger.info(f"    Target per source: {per_source_allocation} chunks")

                    # Track which chunks have been selected (by chunk_id)
                    selected_ids = set()
                    source_shortfalls = []

                    # First pass: allocate per_source_allocation to each source
                    for source in source_names:
                        chunks_in_source = by_source[source]
                        available = len(chunks_in_source)

                        if available >= per_source_allocation:
                            # Take top-ranked chunks up to allocation
                            taken = 0
                            for chunk in chunks_in_source:
                                chunk_id = chunk.get("id") or chunk.get("chunk_id")
                                if chunk_id not in selected_ids and taken < per_source_allocation:
                                    selected.append(chunk)
                                    selected_ids.add(chunk_id)
                                    taken += 1
                            logger.info(f"    Taking {taken} from '{source}' (available: {available})")
                        else:
                            # Source doesn't have enough, take all and track shortfall
                            taken = 0
                            for chunk in chunks_in_source:
                                chunk_id = chunk.get("id") or chunk.get("chunk_id")
                                if chunk_id not in selected_ids:
                                    selected.append(chunk)
                                    selected_ids.add(chunk_id)
                                    taken += 1
                            shortfall = per_source_allocation - taken
                            source_shortfalls.append(shortfall)
                            logger.info(
                                f"    Taking {taken} from '{source}' (available: {available}, shortfall: {shortfall})"
                            )

                    # Second pass: redistribute shortfalls plus remainder evenly across sources
                    total_redistribution = sum(source_shortfalls) + unallocated

                    if total_redistribution > 0:
                        logger.info(f"    Redistributing {total_redistribution} slots evenly across sources")

                        # Count sources that still have available chunks
                        sources_with_capacity = []
                        for source in source_names:
                            available_in_source = sum(
                                1
                                for chunk in by_source[source]
                                if (chunk.get("id") or chunk.get("chunk_id")) not in selected_ids
                            )
                            if available_in_source > 0:
                                sources_with_capacity.append((source, available_in_source))

                        if sources_with_capacity:
                            # Distribute evenly across sources with capacity
                            per_source_redist = total_redistribution // len(sources_with_capacity)
                            remainder = total_redistribution % len(sources_with_capacity)

                            for idx, (source, available) in enumerate(sources_with_capacity):
                                # Give remainder to first few sources
                                allocation = per_source_redist + (1 if idx < remainder else 0)
                                allocation = min(allocation, available)

                                taken = 0
                                for chunk in by_source[source]:
                                    chunk_id = chunk.get("id") or chunk.get("chunk_id")
                                    if chunk_id not in selected_ids and taken < allocation:
                                        selected.append(chunk)
                                        selected_ids.add(chunk_id)
                                        taken += 1

                                if taken > 0:
                                    logger.info(f"    Redistributed {taken} additional from '{source}'")
                                    total_redistribution -= taken

                    remaining_budget = 0

        merged_chunks = selected[:merged_top_k]
        logger.info(f"After merged_top_k limit: {len(merged_chunks)} chunks selected")

    # Batch fetch content for all unique chunks from Redis
    fetch_start = time.time()
    chunk_ids = [chunk.get("id") or chunk.get("chunk_id") for chunk in merged_chunks]

    if text_chunks_storage is None:
        raise ValueError("text_chunks_storage is required for batch content fetch")

    chunks_data_list = await text_chunks_storage.get_by_ids(chunk_ids)

    # Build ID -> content mapping
    chunks_data_map = {}
    for chunk_data in chunks_data_list:
        chunk_id = chunk_data.get("id") or chunk_data.get("chunk_id")
        if chunk_id:
            chunks_data_map[chunk_id] = chunk_data

    # Add content to merged chunks
    for chunk in merged_chunks:
        chunk_id = chunk.get("id") or chunk.get("chunk_id")
        if chunk_id in chunks_data_map:
            chunk["content"] = chunks_data_map[chunk_id].get("content", "")
        else:
            chunk["content"] = ""  # Fallback for missing chunks

    fetch_elapsed = time.time() - fetch_start
    logger.info(f"Batch fetched content for {len(merged_chunks)} chunks in {fetch_elapsed:.3f}s")

    # Process chunks (rerank + token truncation)
    process_start = time.time()
    global_config = {
        "tokenizer": config.get("tokenizer"),
        "MAX_TOTAL_TOKENS": param.max_total_tokens,
        "rerank_processor": config.get("rerank_processor"),
        "rerank_model_func": config.get("rerank_model_func"),
        "min_rerank_score": param.min_rerank_score,
        "rerank_chunk": config.get("rerank_chunk", {}),
    }

    logger.debug(f"Processing chunks: rerank={param.enable_rerank}, max_tokens={param.max_total_tokens}")

    processed_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=merged_chunks,
        query_param=param,
        global_config=global_config,
        source_type="mixed",
        chunk_token_limit=param.max_total_tokens,
        chunks_vdb=chunks_vdb,
        query_expansions=query_expansions,
        detailed_logger=detailed_logger,
    )
    process_elapsed = time.time() - process_start
    logger.info(f"Processed: {len(processed_chunks)} final chunks in {process_elapsed:.3f}s")

    # Log all final chunks with their ranking scores
    if processed_chunks:
        logger.info(f"All {len(processed_chunks)} final chunks with ranking scores:")
        for i, chunk in enumerate(processed_chunks, 1):
            chunk_id = chunk.get("id") or chunk.get("chunk_id", "unknown")
            score = chunk.get("rerank_score", 0.0)
            content = chunk.get("content") or chunk.get("text", "")
            content_preview = content[:100] + "..." if len(content) > 100 else content
            logger.info(f"  [{i}] {chunk_id}: score={score:.4f} | {content_preview}")

        # Detailed logging: log final top-20 chunks
        if detailed_logger:
            import re

            chunks_list = []
            paper_ids = set()
            paper_distribution = {}

            for i, chunk in enumerate(processed_chunks, 1):
                chunk_id = chunk.get("id") or chunk.get("chunk_id", "unknown")
                score = chunk.get("rerank_score", 0.0)

                # Extract content for this chunk
                content = chunk.get("content") or chunk.get("text", "")
                chunk_content_preview = content[:100] + "..." if len(content) > 100 else content

                # Extract paper ID from chunk ID
                match = re.match(r"chunk-([a-f0-9]{40})-(\d+)", chunk_id)
                if match:
                    paper_id = match.group(1)
                    chunk_num = match.group(2)
                    paper_ids.add(paper_id)
                    paper_distribution[paper_id] = paper_distribution.get(paper_id, 0) + 1
                else:
                    paper_id = "unknown"
                    chunk_num = "unknown"

                chunks_list.append(
                    {
                        "rank": i,
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "chunk_num": chunk_num,
                        "score": float(score),
                        "content_preview": chunk_content_preview,
                    }
                )

            detailed_logger.log_final_top20(
                {
                    "chunks": chunks_list,
                    "unique_papers": sorted(list(paper_ids)),
                    "paper_distribution": paper_distribution,
                    "num_chunks": len(chunks_list),
                    "num_papers": len(paper_ids),
                }
            )

    # Build context
    context_start = time.time()
    context = build_query_context(
        chunks=processed_chunks,
        entities=entity_info,
        relationships=relationship_info,
    )
    # Add failed early reranking chunks (for maybe-related section in prompt)
    if failed_early_rerank_chunks:
        context["maybe_related_chunks"] = failed_early_rerank_chunks
        logger.info(f"Added {len(failed_early_rerank_chunks)} failed early reranking chunks to context")
    else:
        context["maybe_related_chunks"] = []
    context_elapsed = time.time() - context_start
    logger.debug(f"Built context in {context_elapsed:.3f}s")

    # Format for LLM
    llm_prompt = build_llm_context(
        query=query,
        context=context,
        mode=mode,
        response_type=param.response_type,
        min_rerank_score=param.min_rerank_score,
        entities_only=entities_only,
    )
    logger.debug(f"LLM prompt length: {len(llm_prompt)} chars")

    # Generate LLM response
    llm_start = time.time()
    response_text_raw = await generate_llm_response(
        prompt=llm_prompt,
        llm_provider=llm_provider,
    )
    llm_elapsed = time.time() - llm_start
    logger.info(f"LLM response (raw): {len(response_text_raw)} chars in {llm_elapsed:.3f}s")

    # Log the full raw response at INFO level for debugging
    logger.info(f"LLM response (raw content):\n{response_text_raw}")

    # Detailed logging: log LLM response
    if detailed_logger:
        detailed_logger.log_llm_response(
            {
                "model": llm_provider.model if hasattr(llm_provider, "model") else "unknown",
                "num_entities": len(entity_info),
                "num_relationships": len(relationship_info),
                "num_chunks": len(processed_chunks),
                "prompt_length": len(llm_prompt),
                "response_raw": response_text_raw,
                "response_length": len(response_text_raw),
                "timing_ms": int(llm_elapsed * 1000),
            }
        )

    # Extract final answer from <thinking> and <answer> tags
    from ...utils.helpers import extract_answer_from_tags

    response_text, thinking = extract_answer_from_tags(response_text_raw)

    if thinking:
        logger.info(f"Extracted answer: {len(response_text)} chars (thinking: {len(thinking)} chars hidden from user)")
        logger.info(f"Final answer:\n{response_text}")
    else:
        logger.debug("No <thinking> tags found, only <answer> extracted")

    # Format references
    references = format_references(processed_chunks)

    logger.info(
        f"Query completed: {len(entity_info)} entities, {len(relationship_info)} relationships, "
        f"{len(processed_chunks)} chunks, {len(references)} references"
    )

    logger.debug(
        f"Timing breakdown: merge={merge_elapsed:.3f}s, process={process_elapsed:.3f}s, "
        f"context={context_elapsed:.3f}s, llm={llm_elapsed:.3f}s"
    )

    return {
        "response": response_text,
        "query": query,
        "mode": mode,
        "chunks": processed_chunks,
        "entities": entity_info,
        "relationships": relationship_info,
        "references": references,
        "metadata": {
            "chunks_vector": len(vector_chunks),
            "chunks_entity": len(entity_chunks),
            "chunks_relationship": len(relation_chunks),
            "chunks_merged": len(merged_chunks),
            "chunks_final": len(processed_chunks),
            "entities": len(entity_info),
            "relationships": len(relationship_info),
        },
        "timing": {
            "merge": merge_elapsed,
            "process": process_elapsed,
            "context": context_elapsed,
            "llm": llm_elapsed,
        },
    }


async def expand_entities_via_neo4j(
    seed_entities: list[dict],
    graph_storage,
    max_hops: int = 1,
    min_paper_support: int = 3,
    max_per_hop: int = 10,
    relationship_types: str = "REGULATES,INTERACTS_WITH,PART_OF",
) -> list[dict]:
    """
    Expand entity set via Neo4j graph traversal.

    Args:
        seed_entities: Initial entities from n-gram matching
        graph_storage: Neo4j storage instance
        max_hops: Number of hops to traverse (0=disabled, 1=single-hop, 2-3=multi-hop)
        min_paper_support: Minimum paper co-occurrences
        max_per_hop: Maximum entities to discover per hop
        relationship_types: Comma-separated relationship types to traverse

    Returns:
        List of discovered entities with metadata
    """
    if max_hops == 0:
        logger.debug("Neo4j entity expansion disabled (max_hops=0)")
        return []

    import time

    start_time = time.time()
    discovered_entities = []
    current_level = {e.get("entity_id") or e.get("entity_name") or e.get("id") for e in seed_entities}
    all_discovered = set(current_level)  # Track all discovered to avoid duplicates

    # Use simple thresholds (percentile feature disabled for now)
    min_weight_threshold = max(1, int(min_paper_support * 10000)) if min_paper_support < 1 else min_paper_support
    max_weight_threshold = 50000  # Fixed max to avoid overly common entities

    logger.info(f"Neo4j entity expansion: {max_hops} hops, weight >= {min_weight_threshold}")

    for hop in range(max_hops):
        if not current_level:
            break

        logger.info(f"  Hop {hop + 1}: Starting with {len(current_level)} entities")

        # Cypher query - aggregate by connection count, then weight
        cypher = """
        MATCH (seed)-[r:DIRECTED]-(related)
        WHERE seed.entity_id IN $seed_ids
          AND NOT related.entity_id IN $already_discovered
          AND r.weight >= $min_support
          AND r.weight <= $max_support
        WITH related,
             count(DISTINCT seed) as seed_connections,
             max(r.weight) as max_weight,
             related.entity_type as type,
             related.description as description
        RETURN related.entity_id as id,
               type,
               description,
               seed_connections,
               max_weight as paper_count
        ORDER BY seed_connections DESC, max_weight DESC
        LIMIT $max_entities
        """

        try:
            async with graph_storage._driver.session(database=graph_storage._database) as session:
                result = await session.run(
                    cypher,
                    seed_ids=list(current_level),
                    already_discovered=list(all_discovered),
                    min_support=min_weight_threshold,
                    max_support=max_weight_threshold,
                    max_entities=max_per_hop,
                )
                records = await result.data()

            # Collect discovered entities
            next_level = set()
            for record in records:
                entity_id = record.get("id")  # This is entity_id from Neo4j (e.g., "GENE:BRCA1")
                discovered_entities.append(
                    {
                        "entity_id": entity_id,  # Use entity_id key (consistent with n-gram)
                        "entity_name": entity_id,  # Also add as entity_name (Milvus field name)
                        "entity_type": record.get("type", "unknown"),
                        "description": record.get("description", ""),
                        "paper_count": record.get("paper_count", 0),
                        "source": f"neo4j_hop_{hop + 1}",
                    }
                )
                next_level.add(entity_id)
                all_discovered.add(entity_id)

            logger.info(f"  Hop {hop + 1}: Discovered {len(next_level)} new entities")

            current_level = next_level

        except Exception as e:
            logger.error(f"  Hop {hop + 1} failed: {e}")
            import traceback

            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            break

    elapsed = time.time() - start_time
    logger.info(f"Neo4j entity expansion: {len(discovered_entities)} entities discovered in {elapsed:.3f}s")

    return discovered_entities
