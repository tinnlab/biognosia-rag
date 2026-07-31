"""
Hybrid KG mode: Combines local + global retrieval.

Pipeline: (local entities + global relationships) -> merge -> chunks
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


async def hybrid_mode(
    query, embedding_manager, storage_dict, llm_provider, config, param, es_client=None, cache_manager=None
):
    """
    HYBRID MODE: Combine local + global results.

    1. Run LOCAL retrieval (entities first)
    2. Run GLOBAL retrieval (relationships first)
    3. Merge entities and relationships via round-robin
    4. Get chunks from merged results
    """
    from ...retrieval.kg_search import (
        find_most_related_edges_from_entities,
        find_most_related_entities_from_relationships,
        get_edge_data,
        get_node_data,
    )
    from ...retrieval.vector_search import get_relationship_vector_context

    logger.info("=== HYBRID MODE: Combining local + global ===")

    # Unpack storage
    entities_vdb = storage_dict["entities_vdb"]
    relationships_vdb = storage_dict.get("relationships_vdb")
    graph_storage = storage_dict["graph_storage"]
    text_chunks_storage = storage_dict["text_chunks_storage"]
    chunk_entity_relation_storage = storage_dict["chunk_entity_relation_storage"]
    chunks_vdb = storage_dict["chunks_vdb"]

    # ===== LOCAL RETRIEVAL =====
    logger.debug("Running local retrieval...")

    # Get local entities
    local_entities, used_ngram = await extract_entities_ngram(query, config, storage_dict, embedding_manager)

    if not used_ngram or not local_entities:
        local_entities = await get_entities_from_vector_search(
            query, embedding_manager, entities_vdb, graph_storage, param.top_k, param.cosine_threshold
        )

    local_entities = local_entities[: param.top_k // 2]  # Take half for local

    # Get relationships from local entities
    local_relationships = []
    if local_entities:
        edge_pairs = await find_most_related_edges_from_entities(
            entity_ids=[e["entity_id"] for e in local_entities],
            graph_storage=graph_storage,
            top_k=param.top_k // 2,
        )
        local_relationships = await get_edge_data(
            edge_pairs=edge_pairs,
            graph_storage=graph_storage,
            top_k=param.top_k // 2,
        )

    # ===== GLOBAL RETRIEVAL =====
    logger.debug("Running global retrieval...")

    # Get global relationships
    if relationships_vdb:
        relationship_search_results = await get_relationship_vector_context(
            query=query,
            embedding_manager=embedding_manager,
            relationships_vdb=relationships_vdb,
            top_k=param.top_k,
            cosine_threshold=param.cosine_threshold,
        )
        global_relationship_ids = [result["id"] for result in relationship_search_results]
        global_relationships = await get_edge_data(
            edge_pairs=global_relationship_ids,
            graph_storage=graph_storage,
            top_k=param.top_k // 2,
        )
    else:
        # Fallback: get relationships from local entities
        local_entity_ids = [e["entity_id"] for e in local_entities]
        edge_pairs = await find_most_related_edges_from_entities(
            entity_ids=local_entity_ids,
            graph_storage=graph_storage,
            top_k=param.top_k,
        )
        global_relationships = await get_edge_data(
            edge_pairs=edge_pairs,
            graph_storage=graph_storage,
            top_k=param.top_k // 2,
        )

    # Get entities from global relationships
    global_entities = []
    if global_relationships:
        global_entity_ids = await find_most_related_entities_from_relationships(
            relationship_info=global_relationships,
            graph_storage=graph_storage,
            top_k=param.top_k // 2,
        )
        global_entities = await get_node_data(
            node_ids=global_entity_ids,
            graph_storage=graph_storage,
            top_k=param.top_k // 2,
        )

    # ===== MERGE RESULTS (round-robin) =====
    logger.debug("Merging local and global results...")

    # Merge entities
    entity_info = []
    seen_entities = set()
    for i in range(max(len(local_entities), len(global_entities))):
        if i < len(local_entities):
            e = local_entities[i]
            if e["entity_id"] not in seen_entities:
                entity_info.append(e)
                seen_entities.add(e["entity_id"])
        if i < len(global_entities):
            e = global_entities[i]
            if e["entity_id"] not in seen_entities:
                entity_info.append(e)
                seen_entities.add(e["entity_id"])

    # Merge relationships
    relationship_info = []
    seen_relationships = set()
    for i in range(max(len(local_relationships), len(global_relationships))):
        if i < len(local_relationships):
            r = local_relationships[i]
            rel_key = (r["src_id"], r["tgt_id"])
            if rel_key not in seen_relationships:
                relationship_info.append(r)
                seen_relationships.add(rel_key)
        if i < len(global_relationships):
            r = global_relationships[i]
            rel_key = (r["src_id"], r["tgt_id"])
            if rel_key not in seen_relationships:
                relationship_info.append(r)
                seen_relationships.add(rel_key)

    logger.info(f"Hybrid: {len(entity_info)} entities, {len(relationship_info)} relationships")

    # Apply entity reranking to filter merged entities by relevance
    entity_info = await apply_entity_reranking(query, entity_info, config)
    logger.info(f"After entity reranking: {len(entity_info)} entities")

    # ===== GET CHUNKS (returns tuple: chunks + failed early reranking chunks) =====
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

    # Merge failed chunks from entity and relation retrieval
    failed_early_rerank_chunks = failed_entity_chunks + failed_relation_chunks
    if failed_early_rerank_chunks:
        logger.info(
            f"Early reranking failures: {len(failed_entity_chunks)} from entities, "
            f"{len(failed_relation_chunks)} from relations (total: {len(failed_early_rerank_chunks)})"
        )

    # ===== PROCESS AND RESPOND =====
    return await process_and_respond(
        query,
        entity_info,
        relationship_info,
        vector_chunks=[],
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        llm_provider=llm_provider,
        config=config,
        param=param,
        mode="hybrid",
        chunks_vdb=chunks_vdb,
        failed_early_rerank_chunks=failed_early_rerank_chunks,
        text_chunks_storage=text_chunks_storage,
    )
