"""
Knowledge graph search functions for RAG query system.

Based on LightRAG operate.py KG search functions.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..storage.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage

logger = logging.getLogger(__name__)


async def get_node_data(
    node_ids: list[str],
    graph_storage: "BaseGraphStorage",
    entities_vdb: "BaseVectorStorage" = None,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """
    Retrieve entity nodes and their metadata.

    Prefers Milvus (clean descriptions) over Neo4j (long gene lists).

    Args:
        node_ids: List of entity IDs
        graph_storage: Graph database storage
        entities_vdb: Milvus entities vector storage (optional, for clean descriptions)
        top_k: Number of top entities to return (sorted by degree)

    Returns:
        List of entity dictionaries with degree information
    """
    if not node_ids:
        return []

    logger.debug(f"Getting node data for {len(node_ids)} nodes")

    try:
        # Step 1: Try Milvus first (has clean description in dynamic fields)
        nodes_from_milvus = {}

        if entities_vdb:
            from pymilvus import Collection, connections

            # Ensure Milvus connection exists — use host/port from the storage config
            if not connections.has_connection("default"):
                milvus_host = entities_vdb.config.get("host", "localhost")
                milvus_port = entities_vdb.config.get("port", 19530)
                connections.connect("default", host=milvus_host, port=milvus_port)

            # Group node_ids by collection
            nodes_by_collection = {}
            for node_id in node_ids:
                # Map entity ID prefix to collection name
                if node_id.startswith("GENE:"):
                    coll_name = "entities_Genes"
                elif node_id.startswith("GO:"):
                    coll_name = "entities_Gene_Ontology"
                elif node_id.startswith("KEGG:"):
                    coll_name = "entities_KEGG_Pathway"
                elif node_id.startswith("DISEASE:") or node_id.startswith("MESH:"):
                    coll_name = "entities_Disease"
                elif node_id.startswith("CHEM:"):
                    coll_name = "entities_Chemical"
                elif node_id.startswith("CELL:"):
                    coll_name = "entities_Cell_Ontology"
                else:
                    continue

                if coll_name not in nodes_by_collection:
                    nodes_by_collection[coll_name] = []
                nodes_by_collection[coll_name].append(node_id)

            # Batch query each collection
            for coll_name, ids_for_collection in nodes_by_collection.items():
                try:
                    collection = Collection(coll_name)

                    # Build IN expression for batch query (use entity_name which has prefix)
                    ids_str = ", ".join([f'"{eid}"' for eid in ids_for_collection])
                    expr = f"entity_name in [{ids_str}]"

                    logger.info(f"Milvus entity lookup: collection={coll_name}, entities={len(ids_for_collection)}")
                    logger.info(f"  Query: {expr[:200]}...")
                    logger.info(f"  IDs: {ids_for_collection[:5]}...")

                    results = collection.query(
                        expr=expr,
                        output_fields=["*"],  # Get all including dynamic fields
                        limit=len(ids_for_collection),
                    )

                    logger.info(f"  Results: {len(results)} entities found")

                    for entity in results:
                        entity_name = entity.get("entity_name")  # Use entity_name (has prefix)
                        description = entity.get("description", entity.get("label", ""))

                        logger.info(f"    {entity_name}: desc='{description[:50]}...'")

                        # Use description field (clean) not content (gene list)
                        nodes_from_milvus[entity_name] = {
                            "entity_type": entity.get(
                                "entity_type", entity_name.split(":")[0] if ":" in entity_name else "Unknown"
                            ),
                            "description": description,
                        }

                    logger.info(f"  Milvus batch: {len(results)} entities from {coll_name}")

                except Exception as e:
                    logger.error(f"Milvus batch query failed for {coll_name}: {e}", exc_info=True)
                    continue

        # Fallback to Neo4j if needed
        if nodes_from_milvus:
            logger.info(f"✓ Got {len(nodes_from_milvus)}/{len(node_ids)} entities from Milvus (clean descriptions)")
            nodes = nodes_from_milvus
        else:
            logger.warning(f"Using Neo4j for entity metadata ({len(node_ids)} entities - Milvus lookup failed)")
            nodes = await graph_storage.get_nodes_batch(node_ids)

        if not nodes:
            logger.warning(f"No nodes found for {len(node_ids)} IDs")
            return []

        # Step 2: Calculate degrees
        degrees = await graph_storage.node_degrees_batch(list(nodes.keys()))

        # Step 3: Combine node data with degrees
        entity_info = []
        for node_id, node_data in nodes.items():
            entity_info.append(
                {
                    "entity_id": node_id,
                    "entity_type": node_data.get("entity_type", ""),
                    "description": node_data.get("description", ""),
                    "degree": degrees.get(node_id, 0),
                }
            )

        # Step 4: Sort by degree (descending) and limit to top_k
        entity_info.sort(key=lambda x: x["degree"], reverse=True)
        entity_info = entity_info[:top_k]

        logger.debug(f"Retrieved {len(entity_info)} entities (top_k={top_k})")

        return entity_info

    except Exception as e:
        logger.error(f"Failed to get node data: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return []


async def find_most_related_edges_from_entities(
    entity_ids: list[str],
    graph_storage: "BaseGraphStorage",
    top_k: int = 30,
) -> list[tuple[str, str]]:
    """
    Find relationships connected to given entities.

    Based on LightRAG operate.py:3443-3497 (_find_most_related_edges_from_entities)

    Args:
        entity_ids: List of entity IDs
        graph_storage: Graph database storage
        top_k: Number of top relationships to return (sorted by degree)

    Returns:
        List of relationship tuples (src_id, tgt_id)
    """
    if not entity_ids:
        return []

    import time

    start_time = time.perf_counter()
    logger.info(f"Finding edges for {len(entity_ids)} entities")

    try:
        # Step 1: Batch retrieve edges
        retrieve_start = time.perf_counter()
        edges_dict = await graph_storage.get_nodes_edges_batch(entity_ids)
        retrieve_elapsed = time.perf_counter() - retrieve_start
        logger.info(f"  Retrieved edges from Neo4j in {retrieve_elapsed:.3f}s")

        # Step 2: Collect all unique edges
        all_edges = set()
        for edges in edges_dict.values():
            if edges:
                all_edges.update(edges)

        if not all_edges:
            logger.warning(f"No edges found for {len(entity_ids)} entities")
            return []

        all_edges = list(all_edges)
        logger.info(f"  Found {len(all_edges)} unique edges across {len(entity_ids)} entities")

        # Step 3: Calculate edge degrees
        degrees_start = time.perf_counter()
        edge_degrees = await graph_storage.edge_degrees_batch(all_edges)
        degrees_elapsed = time.perf_counter() - degrees_start
        logger.info(f"  Calculated edge degrees in {degrees_elapsed:.3f}s")

        # Step 4: Sort by degree and limit to top_k
        edges_with_degrees = [(edge, edge_degrees.get(edge, 0)) for edge in all_edges]
        edges_with_degrees.sort(key=lambda x: x[1], reverse=True)

        top_edges = [edge for edge, _ in edges_with_degrees[:top_k]]

        total_elapsed = time.perf_counter() - start_time
        logger.info(f"Found {len(top_edges)} top relationships (top_k={top_k}) in {total_elapsed:.3f}s")

        return top_edges

    except Exception as e:
        logger.error(f"Failed to find related edges: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return []


async def get_edge_data(
    edge_pairs: list[tuple[str, str]],
    graph_storage: "BaseGraphStorage",
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """
    Retrieve relationship edges and metadata.

    Based on LightRAG operate.py:3662-3716 (_get_edge_data)

    Args:
        edge_pairs: List of (src_id, tgt_id) tuples
        graph_storage: Graph database storage
        top_k: Number of top relationships to return (sorted by degree)

    Returns:
        List of relationship dictionaries with degree information
    """
    if not edge_pairs:
        return []

    import time

    start_time = time.perf_counter()
    logger.info(f"Getting edge data for {len(edge_pairs)} edges")

    try:
        # Step 1: Calculate degrees (batched)
        degrees_start = time.perf_counter()
        edge_degrees = await graph_storage.edge_degrees_batch(edge_pairs)
        degrees_elapsed = time.perf_counter() - degrees_start
        logger.info(f"  Edge degrees calculated in {degrees_elapsed:.3f}s")

        # Step 2: Retrieve ALL edge properties in a single batch query (OPTIMIZED - was N+1 pattern)
        properties_start = time.perf_counter()
        edges_dict = await graph_storage.get_edges_batch(edge_pairs)
        properties_elapsed = time.perf_counter() - properties_start
        logger.info(
            f"  Edge properties retrieved in {properties_elapsed:.3f}s "
            f"({len(edges_dict)}/{len(edge_pairs)} edges found)"
        )

        # Step 3: Build edge info list
        edge_info = []
        for src_id, tgt_id in edge_pairs:
            edge = edges_dict.get((src_id, tgt_id))
            if edge:
                edge_info.append(
                    {
                        "src_id": src_id,
                        "tgt_id": tgt_id,
                        "description": edge.get("description", ""),
                        "keywords": edge.get("keywords", ""),
                        "weight": edge.get("weight", 1),
                        "degree": edge_degrees.get((src_id, tgt_id), 0),
                    }
                )

        # Step 4: Sort by degree and limit to top_k
        edge_info.sort(key=lambda x: x["degree"], reverse=True)
        edge_info = edge_info[:top_k]

        total_elapsed = time.perf_counter() - start_time
        logger.info(f"Retrieved {len(edge_info)} relationships (top_k={top_k}) in {total_elapsed:.3f}s")

        return edge_info

    except Exception as e:
        logger.error(f"Failed to get edge data: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return []


async def find_most_related_entities_from_relationships(
    edge_pairs: list[tuple[str, str]],
    top_k: int = 30,
) -> list[str]:
    """
    Find entities connected by given relationships.

    Based on LightRAG operate.py:3718-3749 (_find_most_related_entities_from_relationships)

    Args:
        edge_pairs: List of (src_id, tgt_id) tuples
        top_k: Number of top entities to return

    Returns:
        List of entity IDs
    """
    if not edge_pairs:
        return []

    logger.debug(f"Finding entities from {len(edge_pairs)} relationships")

    try:
        # Extract unique entity IDs from edges
        entity_ids = set()
        for src_id, tgt_id in edge_pairs:
            entity_ids.add(src_id)
            entity_ids.add(tgt_id)

        entity_ids = list(entity_ids)[:top_k]

        logger.debug(f"Found {len(entity_ids)} unique entities (top_k={top_k})")

        return entity_ids

    except Exception as e:
        logger.error(f"Failed to find entities from relationships: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return []


async def find_related_text_unit_from_entities(
    entity_info: list[dict[str, Any]],
    text_chunks_storage: "BaseKVStorage",
    chunk_entity_relation_storage: "BaseKVStorage",
    chunks_vdb: "BaseVectorStorage",
    num_of_chunks: int,
    kg_chunk_pick_method: str = "WEIGHT",
    max_related_chunks: int = 5,
    min_related_chunks: int = 1,
    query: str | None = None,
    embedding_func: Callable | None = None,
    query_embedding: Any | None = None,
    llm_provider: Any | None = None,
    enable_candidate_filtering: bool = True,
    candidate_top_k: int = 1000,
    num_query_expansions: int = 2,
    min_query_expansions: int | None = None,
    max_query_expansions: int | None = None,
    min_intersection_size: int = 50,
    max_tokens_query_expansion: int = 1000,
    high_similarity_threshold: float = 0.95,
    query_expansions: list[str] | None = None,
    detailed_logger=None,
) -> list[str]:
    """
    Find text chunks associated with entities.

    Based on LightRAG operate.py:3499-3660 (_find_related_text_unit_from_entities)

    Args:
        entity_info: List of entity dictionaries
        text_chunks_storage: KV storage for chunks
        chunk_entity_relation_storage: KV storage for entity->chunks mapping
        chunks_vdb: Vector database for chunks
        num_of_chunks: Total number of chunks to select
        kg_chunk_pick_method: "WEIGHT", "ENTITY_OVERLAP", or "VECTOR"
            - WEIGHT: Linear gradient allocation per entity (entity-centric)
            - ENTITY_OVERLAP: Rank by entity count (chunk-centric, prefers hub chunks)
            - VECTOR: Semantic similarity ranking
        max_related_chunks: Max chunks per high-importance entity (for WEIGHT method)
        min_related_chunks: Min chunks per low-importance entity (for WEIGHT method)
        query: Query text (for VECTOR method)
        embedding_func: Embedding function (for VECTOR method)
        query_embedding: Pre-computed query embedding (for VECTOR method)
        llm_provider: LLM provider for query expansion (for VECTOR method)
        enable_candidate_filtering: Enable query-guided pre-filtering (for VECTOR method)
        candidate_top_k: Candidate pool size for vector search (for VECTOR method)
        num_query_expansions: Number of query expansions (for VECTOR method)
        min_intersection_size: Minimum intersection size to use filtering (for VECTOR method)
        max_tokens_query_expansion: Max tokens for query expansion LLM (for VECTOR method)
        high_similarity_threshold: Chunks with similarity >= this threshold are always included (for VECTOR method)

    Returns:
        Tuple of (chunk_ids, failed_early_rerank_chunks)
        - chunk_ids: List of chunk IDs
        - failed_early_rerank_chunks: List of chunk dicts that failed early reranking threshold
    """
    if not entity_info:
        return [], []

    logger.debug(f"Finding related chunks: {len(entity_info)} entities, method={kg_chunk_pick_method}")

    try:
        # Step 1: Get chunk IDs for each entity from entity_sources Redis keys (BATCH)
        # Redis stores: {workspace}_entity_sources:ent-{hash} → SET of chunk IDs
        # We need to use the HASHED entity ID (from Milvus), not the semantic ID (from Neo4j)

        # Build list of hashed IDs and mapping
        import time

        batch_start = time.time()
        hashed_ids = []
        id_to_entity = {}  # Map hashed_id -> entity dict

        for entity in entity_info:
            hashed_id = entity.get("hashed_id") or entity.get("id")
            semantic_id = entity.get("entity_id", "unknown")

            if not hashed_id:
                logger.debug(f"  Entity {semantic_id}: no hashed_id or id field available")
                entity["sorted_chunks"] = []
                continue

            hashed_ids.append(hashed_id)
            id_to_entity[hashed_id] = entity

        if not hashed_ids:
            logger.warning("No hashed IDs found for any entities")
            return [], []

        # Batch get all chunk mappings from Redis
        logger.debug(f"Batch fetching chunk mappings for {len(hashed_ids)} entities from Redis...")
        chunk_data_list = await chunk_entity_relation_storage.get_by_ids(hashed_ids)
        batch_elapsed = time.time() - batch_start

        logger.info(f"Redis batch fetch: {len(chunk_data_list)}/{len(hashed_ids)} entities in {batch_elapsed:.3f}s")

        # Build mapping: hashed_id -> chunk_data
        chunk_data_map = {}
        for data in chunk_data_list:
            # Get the ID from the data (RedisStorage adds it)
            data_id = data.get("id")
            if data_id:
                chunk_data_map[data_id] = data

        # Parse chunk IDs for each entity
        for hashed_id, entity in id_to_entity.items():
            semantic_id = entity.get("entity_id", "unknown")
            chunk_data = chunk_data_map.get(hashed_id)

            if chunk_data:
                # Parse chunk IDs from Redis SET format
                # RedisStorage returns: {"id": "ent-xxx", "value": ["chunk-1", "chunk-2", ...]}
                if isinstance(chunk_data, dict):
                    value = chunk_data.get("value", [])
                    chunk_ids = value if isinstance(value, list) else []
                elif isinstance(chunk_data, list):
                    chunk_ids = chunk_data
                else:
                    chunk_ids = []

                entity["sorted_chunks"] = chunk_ids
                logger.debug(f"  {semantic_id}: {len(chunk_ids)} chunks")
            else:
                entity["sorted_chunks"] = []
                logger.debug(f"  {semantic_id}: No chunks found in Redis")

        # Step 2: Apply chunk picking strategy
        failed_early_rerank_chunks = []  # Will be populated by VECTOR method

        if kg_chunk_pick_method == "WEIGHT":
            # Use weighted polling
            from .chunk_picking import pick_by_weighted_polling

            chunk_ids = pick_by_weighted_polling(
                entities_or_relations=entity_info,
                max_related_chunks=max_related_chunks,
                min_related_chunks=min_related_chunks,
            )

        elif kg_chunk_pick_method == "ENTITY_OVERLAP":
            # Use entity overlap count (chunk centrality)
            from .chunk_picking import pick_by_entity_overlap

            chunk_ids = await pick_by_entity_overlap(
                entities_or_relations=entity_info,
                text_chunks_storage=text_chunks_storage,
                top_k=num_of_chunks,
                detailed_logger=detailed_logger,
            )

        elif kg_chunk_pick_method == "VECTOR":
            # Use vector similarity with query-guided pre-filtering
            from .chunk_picking import pick_by_vector_similarity

            chunk_ids, failed_early_rerank_chunks = await pick_by_vector_similarity(
                query=query or "",
                text_chunks_storage=text_chunks_storage,
                chunks_vdb=chunks_vdb,
                num_of_chunks=num_of_chunks,
                entity_info=entity_info,
                embedding_func=embedding_func,
                query_embedding=query_embedding,
                llm_provider=llm_provider,
                enable_candidate_filtering=enable_candidate_filtering,
                candidate_top_k=candidate_top_k,
                num_query_expansions=num_query_expansions,
                min_query_expansions=min_query_expansions,
                max_query_expansions=max_query_expansions,
                min_intersection_size=min_intersection_size,
                max_tokens_query_expansion=max_tokens_query_expansion,
                query_expansions=query_expansions,
                high_similarity_threshold=high_similarity_threshold,
            )

        else:
            logger.error(f"Unknown chunk pick method: {kg_chunk_pick_method}")
            chunk_ids = []

        logger.info(
            f"Selected {len(chunk_ids)} chunks using {kg_chunk_pick_method} method (returning top {num_of_chunks})"
        )

        # Detailed logging: log all chunks to retrieval_kg.jsonl
        final_chunk_ids = chunk_ids[:num_of_chunks]
        if detailed_logger:
            # Collect total chunks across all entities for statistics
            total_chunks = 0
            for entity in entity_info:
                total_chunks += len(entity.get("sorted_chunks", []))

            # Log individual chunks (simplified - no entity counting to avoid O(n*m) nested loop)
            for rank, chunk_id in enumerate(final_chunk_ids, start=1):
                detailed_logger.log_retrieval_kg_chunk(
                    {
                        "chunk_id": chunk_id,
                        "rank": rank,
                    }
                )

            # Log summary to retrieval_kg_summary.json
            detailed_logger.log_retrieval_kg_summary(
                {
                    "method": f"entities_{kg_chunk_pick_method}",
                    "entities_used": len(entity_info),
                    "total_chunks": total_chunks,
                    "selected_chunks": len(final_chunk_ids),
                    "timing_ms": int(batch_elapsed * 1000),
                }
            )

        return final_chunk_ids, failed_early_rerank_chunks

    except Exception as e:
        logger.error(f"Failed to find related text units from entities: {e}")
        import traceback

        logger.error(traceback.format_exc())
        # Re-raise to fail the pipeline - chunk retrieval is critical for query quality
        raise


async def find_related_text_unit_from_relations(
    relation_info: list[dict[str, Any]],
    text_chunks_storage: "BaseKVStorage",
    chunk_entity_relation_storage: "BaseKVStorage",
    chunks_vdb: "BaseVectorStorage",
    num_of_chunks: int,
    kg_chunk_pick_method: str = "WEIGHT",
    max_related_chunks: int = 5,
    min_related_chunks: int = 1,
    query: str | None = None,
    embedding_func: Callable | None = None,
    query_embedding: Any | None = None,
    llm_provider: Any | None = None,
    enable_candidate_filtering: bool = True,
    candidate_top_k: int = 1000,
    num_query_expansions: int = 2,
    min_query_expansions: int | None = None,
    max_query_expansions: int | None = None,
    min_intersection_size: int = 50,
    max_tokens_query_expansion: int = 1000,
    high_similarity_threshold: float = 0.95,
    query_expansions: list[str] | None = None,
    detailed_logger=None,
) -> list[str]:
    """
    Find text chunks associated with relationships.

    Based on LightRAG operate.py:3751-3955 (_find_related_text_unit_from_relations)

    Similar to find_related_text_unit_from_entities but for relationships.

    Args:
        relation_info: List of relationship dictionaries
        text_chunks_storage: KV storage for chunks
        chunk_entity_relation_storage: KV storage for relation->chunks mapping
        chunks_vdb: Vector database for chunks
        num_of_chunks: Total number of chunks to select
        kg_chunk_pick_method: "WEIGHT", "ENTITY_OVERLAP", or "VECTOR"
            - WEIGHT: Linear gradient allocation per relationship (entity-centric)
            - ENTITY_OVERLAP: Rank by relationship count (chunk-centric, prefers hub chunks)
            - VECTOR: Semantic similarity ranking
        max_related_chunks: Max chunks per high-importance relationship (for WEIGHT method)
        min_related_chunks: Min chunks per low-importance relationship (for WEIGHT method)
        query: Query text (for VECTOR method)
        embedding_func: Embedding function (for VECTOR method)
        query_embedding: Pre-computed query embedding (for VECTOR method)
        llm_provider: LLM provider for query expansion (for VECTOR method)
        enable_candidate_filtering: Enable query-guided pre-filtering (for VECTOR method)
        candidate_top_k: Candidate pool size for vector search (for VECTOR method)
        num_query_expansions: Number of query expansions (for VECTOR method)
        min_intersection_size: Minimum intersection size to use filtering (for VECTOR method)
        max_tokens_query_expansion: Max tokens for query expansion LLM (for VECTOR method)
        high_similarity_threshold: Chunks with similarity >= this threshold are always included (for VECTOR method)

    Returns:
        Tuple of (chunk_ids, failed_early_rerank_chunks)
        - chunk_ids: List of chunk IDs
        - failed_early_rerank_chunks: List of chunk dicts that failed early reranking threshold
    """
    if not relation_info:
        return [], []

    logger.debug(
        f"Finding related chunks from relationships: {len(relation_info)} relations, method={kg_chunk_pick_method}"
    )

    try:
        # Step 1: Get chunk IDs for each relationship from Redis
        # Note: Relationships currently use individual lookups (not batch optimized)
        for relation in relation_info:
            src_id = relation["src_id"]
            tgt_id = relation["tgt_id"]

            # Relationship ID format: rel-{MD5(src_id<SEP>tgt_id)}
            # Storage might use different key format for relationships
            rel_key = f"{src_id}_{tgt_id}"  # Or use compute_mdhash_id
            rel_data = await chunk_entity_relation_storage.get_by_id(rel_key)

            if rel_data:
                # Parse chunk IDs from Redis SET format (same as entities)
                # RedisStorage returns: {"id": "rel-xxx", "value": ["chunk-1", "chunk-2", ...]}
                if isinstance(rel_data, dict):
                    value = rel_data.get("value", [])
                    chunk_ids = value if isinstance(value, list) else []
                elif isinstance(rel_data, list):
                    chunk_ids = rel_data
                else:
                    chunk_ids = []

                relation["sorted_chunks"] = chunk_ids
            else:
                relation["sorted_chunks"] = []

        # Step 2: Apply chunk picking strategy (same as entities)
        failed_early_rerank_chunks = []  # Will be populated by VECTOR method

        if kg_chunk_pick_method == "WEIGHT":
            from .chunk_picking import pick_by_weighted_polling

            chunk_ids = pick_by_weighted_polling(
                entities_or_relations=relation_info,
                max_related_chunks=max_related_chunks,
                min_related_chunks=min_related_chunks,
            )

        elif kg_chunk_pick_method == "ENTITY_OVERLAP":
            # Use entity overlap count (chunk centrality)
            from .chunk_picking import pick_by_entity_overlap

            chunk_ids = await pick_by_entity_overlap(
                entities_or_relations=relation_info,
                text_chunks_storage=text_chunks_storage,
                top_k=num_of_chunks,
                detailed_logger=detailed_logger,
            )

        elif kg_chunk_pick_method == "VECTOR":
            from .chunk_picking import pick_by_vector_similarity

            chunk_ids, failed_early_rerank_chunks = await pick_by_vector_similarity(
                query=query or "",
                text_chunks_storage=text_chunks_storage,
                chunks_vdb=chunks_vdb,
                num_of_chunks=num_of_chunks,
                entity_info=relation_info,  # Same interface
                embedding_func=embedding_func,
                query_embedding=query_embedding,
                llm_provider=llm_provider,
                enable_candidate_filtering=enable_candidate_filtering,
                candidate_top_k=candidate_top_k,
                num_query_expansions=num_query_expansions,
                min_query_expansions=min_query_expansions,
                max_query_expansions=max_query_expansions,
                query_expansions=query_expansions,
                min_intersection_size=min_intersection_size,
                max_tokens_query_expansion=max_tokens_query_expansion,
                high_similarity_threshold=high_similarity_threshold,
            )

        else:
            logger.error(f"Unknown chunk pick method: {kg_chunk_pick_method}")
            chunk_ids = []

        logger.debug(f"Selected {len(chunk_ids)} chunks from relationships using {kg_chunk_pick_method} method")

        # Detailed logging: log all chunks to retrieval_kg.jsonl
        final_chunk_ids = chunk_ids[:num_of_chunks]
        if detailed_logger:
            # Collect total chunks across all relationships for statistics
            total_chunks = 0
            for relation in relation_info:
                total_chunks += len(relation.get("sorted_chunks", []))

            # Log individual chunks (simplified - no relation counting to avoid O(n*m) nested loop)
            for rank, chunk_id in enumerate(final_chunk_ids, start=1):
                detailed_logger.log_retrieval_kg_chunk(
                    {
                        "chunk_id": chunk_id,
                        "rank": rank,
                    }
                )

            # Log summary to retrieval_kg_summary.json
            detailed_logger.log_retrieval_kg_summary(
                {
                    "method": f"relations_{kg_chunk_pick_method}",
                    "relations_used": len(relation_info),
                    "total_chunks": total_chunks,
                    "selected_chunks": len(final_chunk_ids),
                    "timing_ms": 0,
                }
            )

        return final_chunk_ids, failed_early_rerank_chunks

    except Exception as e:
        logger.error(f"Failed to find related text units from relations: {e}")
        import traceback

        logger.error(traceback.format_exc())
        # Re-raise to fail the pipeline - chunk retrieval is critical for query quality
        raise
