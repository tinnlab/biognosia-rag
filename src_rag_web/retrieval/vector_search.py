"""
Vector search functions for RAG query system.

Based on LightRAG operate.py vector search functions.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..embedding.hf_embedding import EmbeddingManager
    from ..storage.base import BaseKVStorage, BaseVectorStorage

logger = logging.getLogger(__name__)


async def get_vector_context(
    query: str,
    embedding_manager: "EmbeddingManager",
    chunks_vdb: "BaseVectorStorage",
    text_chunks_storage: "BaseKVStorage",
    top_k: int = 10,
    cosine_threshold: float = 0.0,
    llm_provider=None,
    enable_query_expansion: bool = True,
    num_query_expansions: int = 5,
    min_query_expansions: int | None = None,
    max_query_expansions: int | None = None,
    max_tokens_query_expansion: int = 5000,
    return_expansions: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[str]]:
    """
    Retrieve chunks via vector similarity search with optional query expansion.

    Based on LightRAG operate.py:2587-2642 (_get_vector_context)

    Args:
        query: User's query text
        embedding_manager: Embedding manager for computing query embedding
        chunks_vdb: Vector database for chunks
        text_chunks_storage: KV storage for chunk content
        top_k: Number of chunks to retrieve (per query if expansion enabled)
        cosine_threshold: Minimum cosine similarity threshold
        llm_provider: LLM provider for query expansion (optional)
        enable_query_expansion: Whether to enable query expansion (default: True)
        num_query_expansions: Number of query expansions to generate (default: 5)
        max_tokens_query_expansion: Max tokens for query expansion LLM (default: 5000)
        return_expansions: If True, returns tuple (chunks, expanded_queries). Default: False

    Returns:
        If return_expansions=False: List of chunk dictionaries with content
        If return_expansions=True: Tuple of (chunks, expanded_queries list)
    """
    logger.debug(f"Vector search: query='{query[:50]}...', top_k={top_k}")

    try:
        # Step 1: Query expansion (if enabled and LLM available)
        queries_to_search = [query]  # Always include original query
        if enable_query_expansion and llm_provider:
            try:
                from .query_expansion import expand_query_for_retrieval

                expanded_queries = await expand_query_for_retrieval(
                    query=query,
                    llm_provider=llm_provider,
                    num_expansions=num_query_expansions,
                    min_expansions=min_query_expansions,
                    max_expansions=max_query_expansions,
                    enable=True,
                    max_tokens=max_tokens_query_expansion,
                    context="for Milvus vector search",
                )

                # Use expanded queries for search
                queries_to_search = expanded_queries
                logger.info(f"Query expansion (for Milvus semantic search): generated {len(queries_to_search)} queries")
            except Exception as e:
                logger.warning(f"Query expansion failed, falling back to original query: {e}")
                queries_to_search = [query]
        else:
            if not enable_query_expansion:
                logger.debug("Query expansion disabled by configuration")
            elif not llm_provider:
                logger.debug("Query expansion skipped: no LLM provider")

        # Step 2: Compute embeddings for all queries
        query_embeddings = await embedding_manager.embed_chunks(queries_to_search)
        logger.debug(f"Query embeddings computed: {len(query_embeddings)} embeddings")

        # Step 3: Search for each query and collect all results
        all_results = {}  # chunk_id -> {score, result_data}
        top_k_per_query = max(1, top_k // len(queries_to_search)) if len(queries_to_search) > 1 else top_k

        for idx, (q, qemb) in enumerate(zip(queries_to_search, query_embeddings), 1):
            try:
                search_results = await chunks_vdb.query(
                    query_text=q,
                    query_embedding=qemb.tolist() if hasattr(qemb, "tolist") else list(qemb),
                    top_k=top_k_per_query,
                    cosine_threshold=cosine_threshold,
                )

                for result in search_results:
                    chunk_id = result["id"]
                    score = result.get("score", 0.0)
                    # Keep highest score if chunk appears in multiple queries
                    if chunk_id not in all_results or score > all_results[chunk_id]["score"]:
                        all_results[chunk_id] = {"score": score, "result": result}

                logger.debug(f"Query {idx}/{len(queries_to_search)}: found {len(search_results)} chunks")

            except Exception as e:
                logger.warning(f"Vector search failed for query {idx}: {e}")
                continue

        if not all_results:
            logger.warning("Vector search returned no results from any query")
            return []

        logger.debug(f"Vector search found {len(all_results)} unique candidates from {len(queries_to_search)} queries")

        # Step 4: Sort by score and limit to top_k
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]

        # Step 5: Return chunk IDs + scores (content will be fetched later in batch from Redis)
        chunks = []
        for chunk_id, data in sorted_results:
            chunk = {
                "id": chunk_id,
                "score": data["score"],
                "source": "vector_search",
            }
            chunks.append(chunk)

        logger.info(
            f"Vector search retrieved {len(chunks)} chunk IDs (top_k={top_k}, queries={len(queries_to_search)})"
        )

        if return_expansions:
            return chunks, queries_to_search
        else:
            return chunks

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        if return_expansions:
            return [], [query]
        else:
            return []


async def get_entity_vector_context(
    query: str,
    embedding_manager: "EmbeddingManager",
    entities_vdb: "BaseVectorStorage",
    top_k: int = 30,
    cosine_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Retrieve entities via vector similarity search.

    Args:
        query: User's query text
        embedding_manager: Embedding manager for computing query embedding
        entities_vdb: Vector database for entities
        top_k: Number of entities to retrieve
        cosine_threshold: Minimum cosine similarity threshold

    Returns:
        List of entity dictionaries with metadata
    """
    logger.debug(f"Entity vector search: query='{query[:50]}...', top_k={top_k}")

    try:
        # Compute query embedding (use content model to match entity collection dimension=1024)
        # Note: Entity collection uses BAAI/bge-m3 embeddings (1024-dim), not label model (768-dim)
        query_embedding = await embedding_manager.embed_query(query, use_label_model=False)
        logger.debug(f"Query embedding computed for entities: shape={query_embedding.shape}")

        # Query vector database
        search_results = await entities_vdb.query(
            query_text=query,
            query_embedding=query_embedding.tolist(),  # Convert numpy array to list
            top_k=top_k,
            cosine_threshold=cosine_threshold,
        )

        if not search_results:
            logger.warning("Entity vector search returned no results")
            return []

        logger.info(f"Entity vector search found {len(search_results)} entities")

        return search_results

    except Exception as e:
        logger.error(f"Entity vector search failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return []


async def get_relationship_vector_context(
    query: str,
    embedding_manager: "EmbeddingManager",
    relationships_vdb: "BaseVectorStorage",
    top_k: int = 30,
    cosine_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Retrieve relationships via vector similarity search.

    Args:
        query: User's query text
        embedding_manager: Embedding manager for computing query embedding
        relationships_vdb: Vector database for relationships
        top_k: Number of relationships to retrieve
        cosine_threshold: Minimum cosine similarity threshold

    Returns:
        List of relationship dictionaries with metadata
    """
    logger.debug(f"Relationship vector search: query='{query[:50]}...', top_k={top_k}")

    try:
        # Compute query embedding (use content model for relationship matching)
        query_embedding = await embedding_manager.embed_query(query, use_label_model=False)
        logger.debug(f"Query embedding computed for relationships: shape={query_embedding.shape}")

        # Query vector database
        search_results = await relationships_vdb.query(
            query_text=query,
            query_embedding=query_embedding.tolist(),  # Convert numpy array to list
            top_k=top_k,
            cosine_threshold=cosine_threshold,
        )

        if not search_results:
            logger.warning("Relationship vector search returned no results")
            return []

        logger.info(f"Relationship vector search found {len(search_results)} relationships")

        return search_results

    except Exception as e:
        logger.error(f"Relationship vector search failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return []
