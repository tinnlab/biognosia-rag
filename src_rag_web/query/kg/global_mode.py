"""
Global KG mode: Relationship-first retrieval.

Pipeline: relationships -> entities -> chunks
"""

import logging

from .helpers import (
    apply_entity_reranking,
    extract_entities_ngram,
    get_chunks_from_entities,
    get_chunks_from_relations,
    process_and_respond,
)

logger = logging.getLogger(__name__)


async def global_mode(
    query, embedding_manager, storage_dict, llm_provider, config, param, es_client=None, cache_manager=None
):
    """
    GLOBAL MODE: Relationship-first approach.

    1. Get relationships (vector search or from entities)
    2. Find entities from relationships
    3. Get chunks from relationships and entities
    4. Optionally supplement with hybrid search chunks (if enabled)

    Args:
        es_client: Elasticsearch client (optional, for hybrid search)
        cache_manager: Redis cache manager (optional, for keyword caching)
    """
    from ...retrieval.kg_search import (
        find_most_related_edges_from_entities,
        find_most_related_entities_from_relationships,
        get_edge_data,
        get_node_data,
    )
    from ...retrieval.vector_search import get_entity_vector_context, get_relationship_vector_context

    logger.info("=== GLOBAL MODE: Relationship-first retrieval ===")

    # Unpack storage
    entities_vdb = storage_dict["entities_vdb"]
    relationships_vdb = storage_dict.get("relationships_vdb")
    graph_storage = storage_dict["graph_storage"]
    text_chunks_storage = storage_dict["text_chunks_storage"]
    chunk_entity_relation_storage = storage_dict["chunk_entity_relation_storage"]
    chunks_vdb = storage_dict["chunks_vdb"]

    # 1. Get relationships
    if relationships_vdb:
        # Use relationship vector search
        relationship_search_results = await get_relationship_vector_context(
            query=query,
            embedding_manager=embedding_manager,
            relationships_vdb=relationships_vdb,
            top_k=param.top_k * 2,
            cosine_threshold=param.cosine_threshold,
        )
        relationship_ids = [result["id"] for result in relationship_search_results]
        relationship_info = await get_edge_data(
            edge_pairs=relationship_ids,
            graph_storage=graph_storage,
            top_k=param.top_k,
        )
        logger.info(f"Found {len(relationship_ids)} relationships via vector search")
    else:
        # Fallback: get relationships from entities
        logger.warning("No relationships_vdb, using entity-based fallback")

        # Try n-gram first
        entity_info, used_ngram = await extract_entities_ngram(query, config, storage_dict, embedding_manager)

        if used_ngram and entity_info:
            entity_ids = [e["entity_id"] for e in entity_info]
        else:
            # Vector search fallback
            entity_search_results = await get_entity_vector_context(
                query,
                embedding_manager,
                entities_vdb,
                param.top_k,
                param.cosine_threshold,
            )
            entity_ids = [result.get("entity_name") or result.get("id") for result in entity_search_results]

        edge_pairs = await find_most_related_edges_from_entities(
            entity_ids=entity_ids,
            graph_storage=graph_storage,
            top_k=param.top_k * 2,
        )
        relationship_info = await get_edge_data(
            edge_pairs=edge_pairs,
            graph_storage=graph_storage,
            top_k=param.top_k,
        )

    logger.info(f"Retrieved {len(relationship_info)} relationships")

    # 2. Find entities from relationships
    entity_info = []
    if relationship_info:
        entity_ids = await find_most_related_entities_from_relationships(
            relationship_info=relationship_info,
            graph_storage=graph_storage,
            top_k=param.top_k,
        )
        entity_info = await get_node_data(
            node_ids=entity_ids,
            graph_storage=graph_storage,
            top_k=param.top_k,
        )
        logger.info(f"Found {len(entity_info)} entities from relationships")

    # 2.5. Apply entity reranking to filter entities by relevance
    entity_info = await apply_entity_reranking(query, entity_info, config)
    logger.info(f"After entity reranking: {len(entity_info)} entities")

    # 3. Get chunks (returns tuple: chunks + failed early reranking chunks)
    relation_chunks, failed_relation_chunks = await get_chunks_from_relations(
        relationship_info,
        text_chunks_storage,
        chunk_entity_relation_storage,
        chunks_vdb,
        query,
        embedding_manager.embed_chunks,
        param,
        llm_provider,
    )

    entity_chunks, failed_entity_chunks = await get_chunks_from_entities(
        entity_info,
        text_chunks_storage,
        chunk_entity_relation_storage,
        chunks_vdb,
        query,
        embedding_manager.embed_chunks,
        param,
        llm_provider,
    )

    # Merge failed chunks from entity and relation retrieval
    failed_early_rerank_chunks = failed_entity_chunks + failed_relation_chunks
    if failed_early_rerank_chunks:
        logger.info(
            f"Early reranking failures: {len(failed_entity_chunks)} from entities, "
            f"{len(failed_relation_chunks)} from relations (total: {len(failed_early_rerank_chunks)})"
        )

    # 3.5. Optional: Supplement with hybrid search chunks
    vector_chunks = []
    use_hybrid = (
        param.enable_hybrid_search and es_client is not None and config.get("hybrid_search", {}).get("enabled", False)
    )

    if use_hybrid:
        from ...retrieval.elasticsearch_search import get_hybrid_context
        from ...retrieval.keyword_expansion import expand_query_keywords

        logger.debug("Supplementing with hybrid search chunks")

        # Keyword expansion
        keyword_data = await expand_query_keywords(
            query=query,
            llm_provider=llm_provider,
            config=config,
            cache_manager=cache_manager,
        )
        keywords = keyword_data["keyword_string"]

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
        logger.info(
            f"Hybrid search added {len(vector_chunks)} supplementary chunks "
            f"(ES={hybrid_result['es_count']}, Milvus={hybrid_result['milvus_count']})"
        )

    # 4. Process and respond
    return await process_and_respond(
        query,
        entity_info,
        relationship_info,
        vector_chunks=vector_chunks,  # Empty unless hybrid search enabled
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        llm_provider=llm_provider,
        config=config,
        param=param,
        mode="global",
        chunks_vdb=chunks_vdb,
        failed_early_rerank_chunks=failed_early_rerank_chunks,
        text_chunks_storage=text_chunks_storage,
    )
