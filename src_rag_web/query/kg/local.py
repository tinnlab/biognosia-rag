"""
Local KG mode: Entity-first retrieval.

Pipeline: entities -> relationships -> chunks
"""

import logging

from .helpers import (
    apply_entity_reranking,
    extract_entities_ngram,
    get_chunks_from_entities,
    get_chunks_from_relations,
    get_entities_from_vector_search,
    process_and_respond,
)

logger = logging.getLogger(__name__)


async def local_mode(
    query, embedding_manager, storage_dict, llm_provider, config, param, es_client=None, cache_manager=None
):
    """
    LOCAL MODE: Entity-first approach.

    1. Get entities (n-gram or vector search)
    2. Find relationships from entities
    3. Get chunks from entities and relationships
    4. Optionally supplement with hybrid search chunks (if enabled)

    Args:
        es_client: Elasticsearch client (optional, for hybrid search)
        cache_manager: Redis cache manager (optional, for keyword caching)
    """
    from ...retrieval.kg_search import find_most_related_edges_from_entities, get_edge_data

    logger.info("=== LOCAL MODE: Entity-first retrieval ===")

    # Unpack storage
    entities_vdb = storage_dict["entities_vdb"]
    graph_storage = storage_dict["graph_storage"]
    text_chunks_storage = storage_dict["text_chunks_storage"]
    chunk_entity_relation_storage = storage_dict["chunk_entity_relation_storage"]
    chunks_vdb = storage_dict["chunks_vdb"]

    # 1. Get entities (n-gram matching or vector search fallback)
    entity_info, used_ngram = await extract_entities_ngram(query, config, storage_dict, embedding_manager)

    if not used_ngram or not entity_info:
        logger.debug("Using vector search fallback for entities")
        entity_info = await get_entities_from_vector_search(
            query, embedding_manager, entities_vdb, graph_storage, param.top_k, param.cosine_threshold
        )

    logger.info(f"Retrieved {len(entity_info)} entities")

    # 1.5. Apply entity reranking to filter entities by relevance
    entity_info = await apply_entity_reranking(query, entity_info, config)
    logger.info(f"After entity reranking: {len(entity_info)} entities")

    # 2. Find relationships from entities
    relationship_info = []
    if entity_info:
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
        logger.info(f"Found {len(relationship_info)} relationships")

    # 3. Get chunks (returns tuple: chunks + failed early reranking chunks)
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

    logger.info(f"Retrieved {len(entity_chunks)} entity chunks, {len(relation_chunks)} relation chunks")

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
        mode="local",
        chunks_vdb=chunks_vdb,
        failed_early_rerank_chunks=failed_early_rerank_chunks,
        text_chunks_storage=text_chunks_storage,
    )
