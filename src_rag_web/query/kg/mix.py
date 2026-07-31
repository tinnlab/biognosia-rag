"""
Mix KG mode: KG data + vector chunks combined.

Pipeline: PARALLEL retrieval (KG, ES, Milvus) → merge → rerank → respond

Uses persistent worker pool (3 workers) that are reused across all queries.
"""

import logging

from ...retrieval.kg_search import get_node_data
from .helpers import (
    apply_entity_reranking,
    extract_entities_ngram,
    get_entities_from_vector_search,
    process_and_respond,
)

logger = logging.getLogger(__name__)


async def mix_mode(
    query,
    embedding_manager,
    storage_dict,
    llm_provider,
    config,
    param,
    es_client=None,
    cache_manager=None,
    worker_pool=None,
    detailed_logger=None,
    entities_only=False,
):
    """
    MIX MODE: KG data + ES + Milvus chunks (PARALLEL RETRIEVAL).

    Pipeline:
    1. Extract entities + relationships
    2. Submit to 3 PERSISTENT WORKERS (reused across all queries):
       - Worker 1: KG chunks (entities + relationships)
       - Worker 2: ES BM25 chunks (keyword expansion)
       - Worker 3: Milvus semantic chunks (query expansion)
    3. Merge + deduplicate + fetch content
    4. Rerank + filter + respond

    Args:
        worker_pool: RetrievalWorkerPool instance (must be started during app init)
        entities_only: If True, skip Neo4j expansion and chunk retrieval (entities only)
    """
    if entities_only:
        logger.info("=== MIX MODE (ENTITIES ONLY): No Neo4j expansion, no chunks ===")
    else:
        logger.info("=== MIX MODE: Using persistent worker pool for parallel retrieval ===")

    # Unpack storage
    entities_vdb = storage_dict["entities_vdb"]
    graph_storage = storage_dict["graph_storage"]
    text_chunks_storage = storage_dict["text_chunks_storage"]
    chunk_entity_relation_storage = storage_dict["chunk_entity_relation_storage"]
    chunks_vdb = storage_dict["chunks_vdb"]

    # Step 1: Extract entities
    entity_info, used_ngram = await extract_entities_ngram(
        query, config, storage_dict, embedding_manager, detailed_logger
    )

    if not used_ngram or not entity_info:
        logger.debug("Using vector search fallback for entities")
        entity_info = await get_entities_from_vector_search(
            query, embedding_manager, entities_vdb, graph_storage, param.top_k, param.cosine_threshold
        )

    # Separate Stage 1 (keep all) from Stage 2 (rank and filter)
    # N-gram matching now returns metadata dict (not boolean)
    if isinstance(used_ngram, dict):
        stage1_count = used_ngram.get("stage1_count", 0)
    else:
        # Fallback: old behavior (boolean) - assume all are Stage 2
        stage1_count = 0

    stage1_entities = entity_info[:stage1_count] if stage1_count > 0 else []
    stage2_entities = entity_info[stage1_count:] if stage1_count < len(entity_info) else []

    logger.info(f"Entities: Stage 1={len(stage1_entities)} (keep all), Stage 2={len(stage2_entities)} (rank & filter)")

    # Get reranking config (used for Stage 2, Neo4j, and overflow handling)
    rerank_config = config.get("rerank_entity", {})
    min_score = rerank_config.get("min_score", 0.25)

    # Enrich Stage 2 entities with Milvus descriptions for ranking
    stage2_filtered = []
    if stage2_entities:
        # Get descriptions for Stage 2 entities
        stage2_ids = [e.get("entity_name") or e.get("entity_id") for e in stage2_entities]
        enriched = await get_node_data(
            node_ids=stage2_ids,
            graph_storage=storage_dict["graph_storage"],
            entities_vdb=storage_dict["entities_vdb"],
            top_k=len(stage2_ids),
        )

        # Merge descriptions
        desc_map = {e["entity_id"]: e for e in enriched}
        for entity in stage2_entities:
            ent_id = entity.get("entity_name") or entity.get("entity_id")
            if ent_id in desc_map:
                entity["description"] = desc_map[ent_id]["description"]
                entity["entity_type"] = desc_map[ent_id]["entity_type"]

        # Rank Stage 2 with descriptions
        stage2_with_desc = [e for e in stage2_entities if e.get("description")]
        stage2_ranked = await apply_entity_reranking(query, stage2_with_desc, config)

        # Filter by threshold
        stage2_filtered = [e for e in stage2_ranked if e.get("rerank_score", 1.0) >= min_score]

        logger.info(
            f"Stage 2: {len(stage2_entities)} → ranked → {len(stage2_filtered)} passed threshold (>={min_score})"
        )

    # Combine for Neo4j expansion seeds
    seed_entities = stage1_entities + stage2_filtered

    # Neo4j expansion (skip if entities_only mode)
    neo4j_entities = []
    expansion_hops = config.get("query", {}).get("neo4j_entity_expansion_hops", 1)
    if not entities_only and expansion_hops > 0 and seed_entities:
        from .helpers import expand_entities_via_neo4j

        query_config = config.get("query", {})
        discovered = await expand_entities_via_neo4j(
            seed_entities=seed_entities,
            graph_storage=storage_dict["graph_storage"],
            max_hops=expansion_hops,
            min_paper_support=query_config.get("neo4j_expansion_min_weight_percentile", 0.1),
            max_per_hop=query_config.get("neo4j_expansion_max_per_hop", 10),
            relationship_types=query_config.get(
                "neo4j_expansion_relationship_types", "REGULATES,INTERACTS_WITH,PART_OF"
            ),
        )

        if discovered:
            # Enrich Neo4j entities with Milvus descriptions
            neo4j_ids = [e.get("entity_id") or e.get("entity_name") for e in discovered]
            neo4j_enriched = await get_node_data(
                node_ids=neo4j_ids,
                graph_storage=storage_dict["graph_storage"],
                entities_vdb=storage_dict["entities_vdb"],
                top_k=len(neo4j_ids),
            )

            desc_map = {e["entity_id"]: e for e in neo4j_enriched}
            for entity in discovered:
                ent_id = entity.get("entity_id") or entity.get("entity_name")
                if ent_id in desc_map:
                    entity["description"] = desc_map[ent_id]["description"]
                    entity["entity_type"] = desc_map[ent_id]["entity_type"]

            # Rank Neo4j entities
            neo4j_with_desc = [e for e in discovered if e.get("description")]
            neo4j_ranked = await apply_entity_reranking(query, neo4j_with_desc, config)

            # Filter by threshold
            neo4j_filtered = [e for e in neo4j_ranked if e.get("rerank_score", 1.0) >= min_score]

            logger.info(f"Neo4j: {len(discovered)} → ranked → {len(neo4j_filtered)} passed threshold (>={min_score})")
            neo4j_entities = neo4j_filtered

    # Combine all: Stage 1 (all) + Stage 2 (filtered) + Neo4j (filtered)
    entity_info = stage1_entities + stage2_filtered + neo4j_entities

    # Handle overflow
    max_entities = rerank_config.get("max_entities", 200)
    if len(entity_info) > max_entities:
        logger.warning(f"Total entities ({len(entity_info)}) exceeds max ({max_entities})")
        # Keep all Stage 1, take top from Stage 2 + Neo4j combined
        stage2_neo4j_combined = stage2_filtered + neo4j_entities
        stage2_neo4j_sorted = sorted(stage2_neo4j_combined, key=lambda e: e.get("rerank_score", 0), reverse=True)
        stage2_neo4j_top = stage2_neo4j_sorted[: max_entities - len(stage1_entities)]
        entity_info = stage1_entities + stage2_neo4j_top
        logger.info(
            f"Overflow handled: kept {len(stage1_entities)} Stage 1 + top {len(stage2_neo4j_top)} from Stage 2/Neo4j"
        )

    logger.info(
        f"Final entity set: {len(entity_info)} entities "
        f"(S1: {len(stage1_entities)}, S2: {len(stage2_filtered)}, Neo4j: {len(neo4j_entities)})"
    )

    # ENTITIES ONLY MODE: Skip relationships and chunks, return entities directly
    if entities_only:
        logger.info("Entities-only mode: skipping relationships and chunks, generating response...")

        # Generate response with entities only (no relationships, no chunks)
        return await process_and_respond(
            query,
            entity_info,
            relationship_info=[],  # No relationships
            vector_chunks=[],  # No vector chunks
            entity_chunks=[],  # No entity chunks
            relation_chunks=[],  # No relation chunks
            llm_provider=llm_provider,
            config=config,
            param=param,
            mode="mix",
            chunks_vdb=None,
            query_expansions=None,
            failed_early_rerank_chunks=[],
            text_chunks_storage=text_chunks_storage,
            detailed_logger=detailed_logger,
            entities_only=True,
        )

    # Step 2: Get relationships
    relationship_info = []
    if entity_info:
        from ...retrieval.kg_search import find_most_related_edges_from_entities, get_edge_data

        edge_pairs = await find_most_related_edges_from_entities(
            entity_ids=[e["entity_id"] for e in entity_info],
            graph_storage=graph_storage,
            top_k=param.top_k,
        )
        relationship_info = await get_edge_data(
            edge_pairs=edge_pairs,
            graph_storage=graph_storage,
            top_k=param.top_k,
        )

    logger.info(f"KG metadata: {len(entity_info)} entities, {len(relationship_info)} relationships")

    # Step 3: Submit to persistent worker pool
    use_hybrid = (
        param.enable_hybrid_search and es_client is not None and config.get("hybrid_search", {}).get("enabled", False)
    )

    if worker_pool is None:
        raise ValueError(
            "worker_pool is required for mix_mode. Initialize and start RetrievalWorkerPool during app init."
        )

    logger.info(f"Submitting to worker pool (hybrid={'enabled' if use_hybrid else 'disabled'})...")

    # Submit tasks to the 3 persistent workers and wait for results
    (
        (entity_chunk_ids, relation_chunk_ids),
        (es_chunk_ids, expanded_keywords),
        (milvus_chunk_ids, query_expansions),
        decomposed_queries,
    ) = await worker_pool.retrieve_parallel(
        query=query,
        entity_info=entity_info,
        relationship_info=relationship_info,
        text_chunks_storage=text_chunks_storage,
        chunk_entity_relation_storage=chunk_entity_relation_storage,
        chunks_vdb=chunks_vdb,
        embedding_manager=embedding_manager,
        param=param,
        llm_provider=llm_provider,
        es_client=es_client,
        config=config,
        cache_manager=cache_manager,
        use_hybrid=use_hybrid,
        detailed_logger=detailed_logger,
    )

    logger.info("Worker pool retrieval complete!")

    # Log query/keyword expansions
    if detailed_logger:
        if query_expansions:
            # query_expansions might be a dict with metadata or just a list
            if isinstance(query_expansions, dict):
                all_queries = query_expansions.get("all_queries", [])
                expansions = query_expansions.get("expansions", [])
                hyde = query_expansions.get("hyde", [])

                detailed_logger.log_query_expansion(
                    {
                        "original_query": query,
                        "milvus_queries": all_queries,
                        "num_queries": len(all_queries),
                        "query_breakdown": {
                            "expansions": len(expansions) - 1,  # Minus original
                            "hyde": len(hyde),
                        },
                    }
                )
            else:
                # Backward compatibility
                detailed_logger.log_query_expansion(
                    {
                        "original_query": query,
                        "milvus_queries": query_expansions,
                        "num_queries": len(query_expansions),
                    }
                )
        if expanded_keywords:
            detailed_logger.log_keyword_expansion(
                {
                    "keywords": expanded_keywords if isinstance(expanded_keywords, list) else [expanded_keywords],
                    "num_keywords": len(expanded_keywords) if isinstance(expanded_keywords, list) else 1,
                }
            )

    # Step 4: Merge chunk IDs with source tracking
    chunk_id_to_sources = {}  # chunk_id -> set of sources

    # Track entity chunks
    for chunk_id in entity_chunk_ids:
        if chunk_id not in chunk_id_to_sources:
            chunk_id_to_sources[chunk_id] = set()
        chunk_id_to_sources[chunk_id].add("entity")

    # Track relation chunks
    for chunk_id in relation_chunk_ids:
        if chunk_id not in chunk_id_to_sources:
            chunk_id_to_sources[chunk_id] = set()
        chunk_id_to_sources[chunk_id].add("relation")

    # Track ES chunks
    for chunk_id in es_chunk_ids:
        if chunk_id not in chunk_id_to_sources:
            chunk_id_to_sources[chunk_id] = set()
        chunk_id_to_sources[chunk_id].add("elasticsearch")

    # Track Milvus chunks
    for chunk_id in milvus_chunk_ids:
        if chunk_id not in chunk_id_to_sources:
            chunk_id_to_sources[chunk_id] = set()
        chunk_id_to_sources[chunk_id].add("milvus")

    unique_chunk_ids = list(chunk_id_to_sources.keys())

    logger.info(
        f"Merged chunks: {len(unique_chunk_ids)} unique "
        f"(entity={len(entity_chunk_ids)}, relation={len(relation_chunk_ids)}, "
        f"ES={len(es_chunk_ids)}, Milvus={len(milvus_chunk_ids)})"
    )

    # Step 5: Fetch content ONCE for unique chunk IDs
    logger.info(f"Fetching content for {len(unique_chunk_ids)} unique chunks...")
    chunks_data_list = await text_chunks_storage.get_by_ids(unique_chunk_ids)
    chunks_data = {chunk.get("id") or chunk.get("chunk_id"): chunk for chunk in chunks_data_list}

    # Build chunk lists with source annotations
    entity_chunks = []
    relation_chunks = []
    es_chunks = []
    milvus_chunks = []

    for chunk_id, sources in chunk_id_to_sources.items():
        chunk_data = chunks_data.get(chunk_id)
        if not chunk_data:
            continue

        chunk_obj = {
            "id": chunk_id,
            "content": chunk_data.get("content", ""),
            "sources": list(sources),  # Track all sources
        }

        # Add to appropriate lists (chunk may appear in multiple)
        if "entity" in sources:
            entity_chunks.append(chunk_obj)
        if "relation" in sources:
            relation_chunks.append(chunk_obj)
        if "elasticsearch" in sources:
            es_chunks.append(chunk_obj)
        if "milvus" in sources:
            milvus_chunks.append(chunk_obj)

    # Step 6: Calculate and log chunk source intersections
    entity_chunk_ids_set = set(entity_chunk_ids) | set(relation_chunk_ids)  # Combine entity + relation
    es_chunk_ids_set = set(es_chunk_ids)
    milvus_chunk_ids_set = set(milvus_chunk_ids)

    # Calculate intersections
    entity_and_es = entity_chunk_ids_set & es_chunk_ids_set
    entity_and_milvus = entity_chunk_ids_set & milvus_chunk_ids_set
    es_and_milvus = es_chunk_ids_set & milvus_chunk_ids_set
    all_three = entity_chunk_ids_set & es_chunk_ids_set & milvus_chunk_ids_set

    logger.info("=== CHUNK SOURCE INTERSECTIONS ===")
    logger.info(f"Total unique chunks: {len(unique_chunk_ids)}")
    logger.info(f"  - Entities (entity + relation): {len(entity_chunk_ids_set)} chunks")
    logger.info(f"  - Elasticsearch: {len(es_chunk_ids_set)} chunks")
    logger.info(f"  - Milvus Semantic: {len(milvus_chunk_ids_set)} chunks")
    logger.info("Intersections:")
    logger.info(f"  - Entities ∩ Elasticsearch: {len(entity_and_es)} chunks")
    logger.info(f"  - Entities ∩ Milvus Semantic: {len(entity_and_milvus)} chunks")
    logger.info(f"  - Elasticsearch ∩ Milvus Semantic: {len(es_and_milvus)} chunks")
    logger.info(f"  - All three sources: {len(all_three)} chunks")

    # Step 7: Log top chunks per source
    logger.info("=== TOP CHUNKS PER SOURCE ===")

    if entity_chunks:
        logger.info(f"Entity chunks (top 3 of {len(entity_chunks)}):")
        for i, chunk in enumerate(entity_chunks[:3], 1):
            logger.info(f"  {i}. {chunk['id']}: {chunk['content'][:100]}...")

    if relation_chunks:
        logger.info(f"Relation chunks (top 3 of {len(relation_chunks)}):")
        for i, chunk in enumerate(relation_chunks[:3], 1):
            logger.info(f"  {i}. {chunk['id']}: {chunk['content'][:100]}...")

    if es_chunks:
        logger.info(f"Elasticsearch chunks (top 3 of {len(es_chunks)}):")
        for i, chunk in enumerate(es_chunks[:3], 1):
            logger.info(f"  {i}. {chunk['id']}: {chunk['content'][:100]}...")

    if milvus_chunks:
        logger.info(f"Milvus chunks (top 3 of {len(milvus_chunks)}):")
        for i, chunk in enumerate(milvus_chunks[:3], 1):
            logger.info(f"  {i}. {chunk['id']}: {chunk['content'][:100]}...")

    # Combine ES + Milvus as "vector_chunks" for backward compatibility
    vector_chunks = []
    seen_vector_ids = set()
    for chunk in es_chunks + milvus_chunks:
        if chunk["id"] not in seen_vector_ids:
            vector_chunks.append(chunk)
            seen_vector_ids.add(chunk["id"])

    # Log merge statistics
    if detailed_logger:
        detailed_logger.log_merge_statistics(
            {
                "total_unique": len(unique_chunk_ids),
                "sources": {
                    "entity": len(entity_chunk_ids),
                    "relation": len(relation_chunk_ids),
                    "elasticsearch": len(es_chunk_ids),
                    "milvus": len(milvus_chunk_ids),
                },
                "intersections": {
                    "entity_elasticsearch": len(entity_and_es),
                    "entity_milvus": len(entity_and_milvus),
                    "elasticsearch_milvus": len(es_and_milvus),
                    "all_three": len(all_three),
                },
            }
        )

    # Step 8: Process and respond (existing reranking + LLM generation)
    # Determine which queries to use for reranking:
    # - If decomposition enabled: use decomposed queries (includes original)
    # - If decomposition disabled: use only original query for reranking
    # NOTE: query_expansions is a metadata dict from Milvus worker, not a query list
    if decomposed_queries:
        queries_for_reranking = decomposed_queries
    else:
        # No decomposition - use only original query for reranking
        # (expansions and HyDE are only for retrieval, not reranking)
        queries_for_reranking = [query]

    return await process_and_respond(
        query,
        entity_info,
        relationship_info,
        vector_chunks=vector_chunks,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        llm_provider=llm_provider,
        config=config,
        param=param,
        mode="mix",
        chunks_vdb=chunks_vdb,
        query_expansions=queries_for_reranking,  # For reranking
        failed_early_rerank_chunks=[],  # No early reranking with WEIGHT method
        text_chunks_storage=text_chunks_storage,
        detailed_logger=detailed_logger,
    )
