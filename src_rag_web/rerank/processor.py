"""
Reranking processor for applying rerank models to retrieved documents.

Adapted from plans/lightrag-code/rerank/processor.py
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


async def expand_documents(
    filtered_chunks: list[dict[str, Any]],
    chunks_vdb,
    query: str,
    rerank_processor,
    min_score: float = 0.0,
    query_expansions: list[str] | None = None,
    rerank_chunk_config: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Expand document context by retrieving all chunks from source documents.

    After initial reranking filter, this function:
    1. Extracts unique document IDs from passing chunks
    2. Queries all chunks from those documents using full_doc_id
    3. Reranks all expanded chunks (with multi-query if expansions provided)
    4. Filters with min rerank score
    5. Sorts by chunk ID

    Args:
        filtered_chunks: Chunks that passed initial reranking filter
        chunks_vdb: Milvus vector storage for chunks
        query: User's query for reranking
        rerank_processor: RerankProcessor instance
        min_score: Minimum rerank score threshold
        query_expansions: List of expanded queries for multi-query reranking (optional)
        rerank_chunk_config: Chunk reranking configuration (optional)

    Returns:
        Expanded and reranked list of chunks sorted by ID
    """
    import time

    if not filtered_chunks:
        logger.debug("Document expansion: no chunks to expand")
        return filtered_chunks

    start_time = time.time()

    # Step 1: Extract unique document IDs from chunk IDs
    # Chunk ID format: chunk-{doc_id}-{chunk_num}
    doc_ids = set()
    for chunk in filtered_chunks:
        chunk_id = chunk.get("id") or chunk.get("chunk_id", "")
        if not chunk_id:
            continue

        # Split by '-' and get middle part (document ID)
        parts = chunk_id.split("-")
        if len(parts) >= 3:
            doc_id = parts[1]
            doc_ids.add(doc_id)

    if not doc_ids:
        logger.warning("Document expansion: could not extract any document IDs from chunks")
        return filtered_chunks

    logger.info(f"Document expansion: {len(doc_ids)} unique documents from {len(filtered_chunks)} chunks")

    # Step 2: Query Milvus for all chunks from these documents
    # Build filter: full_doc_id in ["doc1", "doc2", ...]
    try:
        doc_id_list = sorted(doc_ids)
        doc_id_str = ", ".join([f'"{doc_id}"' for doc_id in doc_id_list])
        filter_expr = f"full_doc_id in [{doc_id_str}]"

        logger.info(f"Querying chunks with filter: full_doc_id in [{len(doc_id_list)} documents]")
        query_start = time.time()

        # Query using Milvus client directly (no vector search needed)
        expanded_results = chunks_vdb._client.query(
            collection_name=chunks_vdb.full_collection_name,
            filter=filter_expr,
            output_fields=["*"],
            limit=10000,
        )

        query_elapsed = time.time() - query_start
        logger.info(f"Document expansion: retrieved {len(expanded_results)} chunks in {query_elapsed:.3f}s")

        if not expanded_results:
            logger.warning("Document expansion: no chunks found for document IDs")
            return filtered_chunks

        # Step 3: Build map of original chunks with their scores (to avoid re-ranking)
        original_chunk_map = {(chunk.get("id") or chunk.get("chunk_id")): chunk for chunk in filtered_chunks}
        original_chunk_ids = set(original_chunk_map.keys())

        # Separate expanded chunks into those that need reranking vs those that already have scores
        chunks_with_scores = []  # Already have rerank scores from original pass
        chunks_need_rerank = []  # New chunks from expansion

        for chunk in expanded_results:
            chunk_id = chunk.get("id")
            if chunk_id in original_chunk_ids:
                # This chunk already has a rerank score - preserve it
                chunks_with_scores.append(original_chunk_map[chunk_id])
            else:
                # This is a new chunk - needs reranking
                doc_dict = {
                    "id": chunk_id,
                    "content": chunk.get("content", ""),
                    "text": chunk.get("content", ""),
                }
                chunks_need_rerank.append(doc_dict)

        logger.info(
            f"Document expansion: {len(chunks_with_scores)} chunks already have scores, "
            f"{len(chunks_need_rerank)} new chunks need reranking"
        )

        # Step 4: Only rerank NEW chunks (skip chunks that already have scores)
        reranked_new_chunks = []
        if chunks_need_rerank:
            logger.info(f"Document expansion: reranking {len(chunks_need_rerank)} new chunks")
            rerank_start = time.time()

            # Use multi-query reranking if expansions are provided
            if query_expansions and len(query_expansions) > 1:
                score_aggregation = (
                    rerank_chunk_config.get("score_aggregation", "max") if rerank_chunk_config else "max"
                )

                logger.info(
                    f"Document expansion: using multi-query reranking with {len(query_expansions)} queries "
                    f"(aggregation={score_aggregation})"
                )
                reranked_new_chunks = await rerank_processor.rerank_multi_query(
                    queries=query_expansions,
                    documents=chunks_need_rerank,
                    top_k=None,
                    score_aggregation=score_aggregation,
                )
            else:
                # Single-query reranking (original behavior)
                reranked_new_chunks = await rerank_processor.rerank(
                    query=query,
                    documents=chunks_need_rerank,
                    top_k=None,
                )

            rerank_elapsed = time.time() - rerank_start
            logger.info(f"Document expansion: reranked {len(reranked_new_chunks)} new chunks in {rerank_elapsed:.3f}s")
        else:
            logger.info("Document expansion: no new chunks to rerank (all chunks already scored)")

        # Log top 10 newly reranked chunks with scores
        if reranked_new_chunks:
            top_10 = reranked_new_chunks[:10]
            logger.info(f"Top {len(top_10)} newly reranked chunks from document expansion:")
            for i, doc in enumerate(top_10, 1):
                chunk_id = doc.get("id") or doc.get("chunk_id", "unknown")
                score = doc.get("rerank_score", 0.0)
                content = doc.get("content") or doc.get("text", "")
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"  [{i}] {chunk_id}: score={score:.4f} | {content_preview}")

        # Step 5: Preserve original chunks + add new chunks that pass threshold
        # Note: original_chunk_map and original_chunk_ids already built at step 3 (lines 99-102)

        # Merge strategy:
        # 1. Keep ALL original chunks (they already passed reranking threshold)
        # 2. Add new chunks from expansion that pass the threshold
        merged_chunks = {}

        # Add original chunks first (preserve their original scores)
        for chunk_id, chunk in original_chunk_map.items():
            merged_chunks[chunk_id] = chunk

        # Add NEW chunks from expansion (only if they pass threshold)
        new_chunks_added = 0
        for doc in reranked_new_chunks:
            chunk_id = doc.get("id") or doc.get("chunk_id")
            # This is a new chunk from document expansion (not in original set)
            if min_score > 0.0:
                if doc.get("rerank_score", 0.0) >= min_score:
                    merged_chunks[chunk_id] = doc
                    new_chunks_added += 1
            else:
                merged_chunks[chunk_id] = doc
                new_chunks_added += 1

        filtered_expanded = list(merged_chunks.values())

        logger.info(
            f"Document expansion: preserved {len(original_chunk_ids)} original chunks, "
            f"added {new_chunks_added} new chunks (total: {len(filtered_expanded)})"
        )

        # Step 6: Sort by rerank score (descending)
        filtered_expanded.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        total_elapsed = time.time() - start_time
        logger.info(
            f"Document expansion: completed in {total_elapsed:.3f}s "
            f"({len(filtered_chunks)} original -> {len(filtered_expanded)} total chunks)"
        )

        return filtered_expanded

    except Exception as e:
        logger.error(f"Document expansion failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.warning("Falling back to original filtered chunks")
        return filtered_chunks


async def apply_rerank(
    query: str, retrieved_docs: list[dict[str, Any]], rerank_func: Callable, top_n: int = None, min_score: float = 0.0
) -> list[dict[str, Any]]:
    """
    Apply reranking to retrieved documents.

    Args:
        query: The search query
        retrieved_docs: List of retrieved documents (dicts with content field)
        rerank_func: Async reranking function
        top_n: Number of top documents to return after reranking
        min_score: Minimum relevance score threshold (filter below this)

    Returns:
        Reranked documents sorted by relevance score
    """
    if not retrieved_docs:
        logger.debug("No documents to rerank")
        return []

    if not rerank_func:
        logger.warning("No rerank function provided, returning original docs")
        return retrieved_docs

    try:
        # Extract document content for reranking
        document_texts = []
        for doc in retrieved_docs:
            # Try multiple possible content fields
            content = (
                doc.get("content") or doc.get("text") or doc.get("chunk_content") or doc.get("document") or str(doc)
            )
            document_texts.append(content)

        # Call rerank function
        # Expected return: List[tuple[int, float]] - [(index, score), ...]
        rerank_results = await rerank_func(query=query, documents=document_texts, top_k=top_n)

        if not rerank_results:
            logger.warning("Rerank returned empty results, using original chunks")
            return retrieved_docs

        # Process rerank results
        reranked_docs = []
        for index, relevance_score in rerank_results:
            # Validate index
            if not (0 <= index < len(retrieved_docs)):
                logger.warning(f"Invalid index {index} from reranker, skipping")
                continue

            # Filter by minimum score
            if relevance_score < min_score:
                logger.debug(f"Filtering doc {index} with score {relevance_score} < {min_score}")
                continue

            # Get original document and add rerank score
            doc = retrieved_docs[index].copy()
            doc["rerank_score"] = relevance_score
            reranked_docs.append(doc)

        logger.info(
            f"Reranked: {len(reranked_docs)} chunks from {len(retrieved_docs)} original "
            f"(min_score={min_score}, top_n={top_n})"
        )

        return reranked_docs

    except Exception as e:
        logger.error(f"CRITICAL: Reranking failed: {e}")
        logger.error("Stopping query pipeline due to reranking failure")
        raise


async def apply_rerank_if_enabled(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    global_config: dict[str, Any],
    enable_rerank: bool = True,
    top_n: int = None,
    chunks_vdb=None,
    query_expansions: list[str] | None = None,
    detailed_logger=None,
) -> list[dict[str, Any]]:
    """
    Apply reranking to retrieved documents if rerank is enabled.

    Supports both RerankProcessor (preferred) and legacy function-based approach.
    After reranking filter, expands document context by retrieving all chunks from source documents.

    NEW: If query_expansions is provided, uses multi-query reranking to compare each chunk
    against all queries and take the best score. This improves recall by matching chunks
    against different semantic aspects of the question.

    Args:
        query: The search query
        retrieved_docs: List of retrieved documents
        global_config: Global configuration containing rerank settings
        enable_rerank: Whether to enable reranking from query parameter
        top_n: Number of top documents to return after reranking
        chunks_vdb: Vector storage for chunks (needed for document expansion)
        query_expansions: List of expanded queries (optional).
            If provided and multi-query is enabled, uses multi-query reranking.
            Format: [original_query, expansion_1, expansion_2, ...]

    Returns:
        Reranked and expanded documents if rerank is enabled, otherwise original documents
    """
    if not enable_rerank or not retrieved_docs:
        logger.debug(f"Rerank disabled or no docs (enable={enable_rerank}, docs={len(retrieved_docs)})")
        return retrieved_docs

    # Check for RerankProcessor (preferred)
    rerank_processor = global_config.get("rerank_processor")
    if rerank_processor:
        # Prepare documents with "text" field (RerankProcessor expects this)
        docs_with_text = []
        for doc in retrieved_docs:
            doc_copy = doc.copy()
            # Ensure "text" field exists (processor.py already handles multiple field names)
            if "text" not in doc_copy:
                doc_copy["text"] = doc.get("content") or doc.get("chunk_content") or doc.get("document") or str(doc)
            docs_with_text.append(doc_copy)

        # Get reranking config
        rerank_chunk_config = global_config.get("rerank_chunk", {})
        score_aggregation = rerank_chunk_config.get("score_aggregation", "max")

        # Extract actual queries from dict if needed (Milvus worker returns dict with metadata)
        actual_queries = query_expansions
        if isinstance(query_expansions, dict):
            actual_queries = query_expansions.get("all_queries", [])

        # Use multi-query reranking if expansions are provided
        using_multi_query = actual_queries and len(actual_queries) > 1

        if using_multi_query:
            logger.info(f"Using multi-query reranking with {len(actual_queries)} queries")
            # Get ALL reranked results (no top_k limit yet - we'll filter by threshold first)
            reranked_docs = await rerank_processor.rerank_multi_query(
                queries=actual_queries,
                documents=docs_with_text,
                top_k=None,  # Don't limit yet - filter by threshold first
                score_aggregation=score_aggregation,
                detailed_logger=detailed_logger,
            )
            # Note: Worker pool already logged top chunks with detailed score breakdown
        else:
            # Single-query reranking (original behavior)
            if not query_expansions:
                logger.debug("No query expansions provided, using single-query reranking")
            else:
                logger.debug("Only one query available, using single-query reranking")

            # Get ALL reranked results (no top_k limit yet)
            reranked_docs = await rerank_processor.rerank(
                query=query,
                documents=docs_with_text,
                top_k=None,  # Don't limit yet - filter by threshold first
            )

        # Log rerank scores for top chunks (only for single-query to avoid duplication with worker pool)
        if not using_multi_query:
            logger.info(f"Chunk reranking: reranked {len(reranked_docs)} chunks")
            if reranked_docs:
                # Log top 20 chunks with scores
                top_chunks = reranked_docs[:20]
                logger.info(f"Top {len(top_chunks)} reranked chunks:")
                for i, doc in enumerate(top_chunks, 1):
                    chunk_id = doc.get("id") or doc.get("chunk_id", "unknown")
                    score = doc.get("rerank_score", 0.0)
                    # Truncate content for logging
                    content = doc.get("content") or doc.get("text", "")
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    logger.info(f"  [{i}] {chunk_id}: score={score:.4f} | {content_preview}")

        # Apply min score filter if configured
        min_score = global_config.get("min_rerank_score", 0.0)
        if min_score > 0.0:
            filtered_docs = []
            for doc in reranked_docs:
                if doc.get("rerank_score", 0.0) >= min_score:
                    filtered_docs.append(doc)

            logger.info(
                f"Chunk reranking: kept {len(filtered_docs)}/{len(reranked_docs)} chunks "
                f"after score filter (min_score={min_score})"
            )

            # Document expansion: expand context using ALL chunks that passed threshold
            if chunks_vdb and filtered_docs:
                filtered_docs = await expand_documents(
                    filtered_chunks=filtered_docs,
                    chunks_vdb=chunks_vdb,
                    query=query,
                    rerank_processor=rerank_processor,
                    min_score=min_score,
                    query_expansions=query_expansions,
                    rerank_chunk_config=rerank_chunk_config,
                )

            # Apply top_n limit AFTER expansion (if specified)
            if top_n is not None and len(filtered_docs) > top_n:
                logger.info(f"Limiting to top {top_n} chunks after expansion (from {len(filtered_docs)} total)")
                filtered_docs = filtered_docs[:top_n]

            return filtered_docs

        # No min score filter, but still expand documents if chunks_vdb is available
        if chunks_vdb and reranked_docs:
            reranked_docs = await expand_documents(
                filtered_chunks=reranked_docs,
                chunks_vdb=chunks_vdb,
                query=query,
                rerank_processor=rerank_processor,
                min_score=0.0,
                query_expansions=query_expansions,
                rerank_chunk_config=rerank_chunk_config,
            )

        # Apply top_n limit AFTER expansion (if specified)
        if top_n is not None and len(reranked_docs) > top_n:
            logger.info(f"Limiting to top {top_n} chunks after expansion (from {len(reranked_docs)} total)")
            reranked_docs = reranked_docs[:top_n]

        return reranked_docs

    # Fallback to legacy function-based approach
    rerank_func = global_config.get("rerank_model_func")
    if not rerank_func:
        logger.warning(
            "Rerank is enabled but no rerank processor or function is configured. "
            "Set up a rerank model or set enable_rerank=False."
        )
        return retrieved_docs

    # Get min score threshold
    min_score = global_config.get("min_rerank_score", 0.0)

    return await apply_rerank(
        query=query, retrieved_docs=retrieved_docs, rerank_func=rerank_func, top_n=top_n, min_score=min_score
    )


class RerankProcessor:
    """
    High-level reranking processor for test and production use.

    This class wraps LocalReranker to provide a dict-based interface
    that's more convenient for integration tests and production code.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        worker_pool: Any | None = None,
        two_stage_pool: Any | None = None,
        two_stage_config: dict[str, Any] | None = None,
    ):
        """
        Initialize rerank processor.

        Args:
            config: Reranking configuration dict with keys:
                - provider: "local", "jina", "cohere", or "aliyun"
                - model: Model name (provider-specific)
                - device: Device for local models ("cuda:0", "cpu")
                - max_length: Max sequence length for local models
                - batch_size: Batch size for local models
                - num_workers: Number of parallel workers (for worker pool)
            worker_pool: Optional RerankWorkerPool instance for parallel reranking
            two_stage_pool: Optional TwoStageRerankerPool instance for cascade reranking
            two_stage_config: Two-stage reranking configuration
        """
        self.config = config or {}
        self.provider = self.config.get("provider", "local")
        self.reranker = None
        self.worker_pool = worker_pool
        self.two_stage_pool = two_stage_pool
        self.two_stage_config = two_stage_config or {}
        self._initialized = False

    async def initialize(self):
        """Initialize the reranker (loads model for local provider)."""
        if self._initialized:
            return

        provider = self.provider

        if provider == "local":
            from .local_reranker import LocalReranker

            model = self.config.get("model", "BAAI/bge-reranker-v2-m3")
            device = self.config.get("device", "cuda:0")
            max_length = self.config.get("max_length", 512)
            batch_size = self.config.get("batch_size", 32)
            normalize = self.config.get("normalize", True)

            self.reranker = LocalReranker(
                model_name=model,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
                normalize=normalize,
            )
            self.reranker.initialize()

        elif provider in ["jina", "cohere", "aliyun"]:
            # API-based rerankers don't need initialization
            logger.info(f"Using API-based reranker: {provider}")
        else:
            raise ValueError(f"Unknown rerank provider: {provider}")

        self._initialized = True
        logger.info(f"RerankProcessor initialized with provider: {provider}")

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of document dicts, each must have "text" key
            top_k: Number of top results to return (None = all)

        Returns:
            List of document dicts with added "rerank_score" field,
            sorted by relevance (highest first)
        """
        if not self._initialized:
            await self.initialize()

        if not documents:
            return []

        # Extract texts from document dicts
        texts = [doc.get("text", "") for doc in documents]

        if self.provider == "local":
            # Use local reranker
            if self.reranker is None:
                raise RuntimeError("Local reranker not initialized")

            # Get (index, score) tuples
            indexed_scores = await self.reranker.rerank(query, texts, top_k=top_k)

            # Map back to original documents with scores
            reranked_docs = []
            for idx, score in indexed_scores:
                doc = documents[idx].copy()  # Copy to avoid modifying original
                doc["rerank_score"] = score
                reranked_docs.append(doc)

            return reranked_docs

        else:
            # For API-based rerankers, use apply_rerank with appropriate function
            raise NotImplementedError(f"API-based reranking not yet implemented in RerankProcessor: {self.provider}")

    async def rerank_multi_query(
        self,
        queries: list[str],
        documents: list[dict[str, Any]],
        top_k: int | None = None,
        score_aggregation: str = "max",
        detailed_logger=None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents against multiple queries in parallel.

        If worker_pool is available, distributes work across workers for maximum parallelism.
        Otherwise, falls back to sequential reranking.

        Computes ALL query×document pairs (no early stopping), then aggregates using
        max or mean across queries.

        Args:
            queries: List of query strings (original + expansions)
            documents: List of document dicts with "text" field
            top_k: Number of top results to return (None = all)
            score_aggregation: How to combine scores across queries
                - "max": Take highest score across all queries (DEFAULT)
                - "mean": Average score across all queries

        Returns:
            List of document dicts with added fields:
                - "rerank_score": Final aggregated score
                - "rerank_best_query_idx": Index of query with highest score
                - "rerank_query_scores": Dict mapping query index -> score
            Sorted by aggregated relevance (highest first)
        """
        import time

        if not queries or not documents:
            return []

        if len(queries) == 1:
            logger.debug("Single query provided to multi-query reranking, using standard rerank")
            return await self.rerank(query=queries[0], documents=documents, top_k=top_k)

        start_time = time.time()
        num_docs = len(documents)
        num_queries = len(queries)

        # Use two-stage pool if available and enabled for cascade reranking
        if self.two_stage_pool is not None and self.two_stage_config.get("enabled", False):
            logger.info(
                f"Using TWO-STAGE reranking: {num_queries} queries × {num_docs} documents "
                f"(Stage 1 filter → top {self.two_stage_config.get('stage1_top_k', 10000)} → Stage 2)"
            )
            aggregated_results = await self.two_stage_pool.rerank_two_stage(
                queries=queries,
                documents=documents,
                score_aggregation=score_aggregation,
                detailed_logger=detailed_logger,
            )

            # Apply top_k if specified
            if top_k is not None:
                aggregated_results = aggregated_results[:top_k]

            return aggregated_results

        # Use single-stage worker pool if available for parallel reranking
        if self.worker_pool is not None:
            logger.info(
                f"Using rerank worker pool: {num_queries} queries × {num_docs} documents = "
                f"{num_queries * num_docs} pairs in parallel"
            )
            aggregated_results = await self.worker_pool.rerank_all_pairs(
                queries=queries,
                documents=documents,
                score_aggregation=score_aggregation,
            )

            # Apply top_k if specified
            if top_k is not None:
                aggregated_results = aggregated_results[:top_k]

            return aggregated_results

        # Fall back to sequential reranking (old behavior, but without early stopping)
        logger.info(
            f"No worker pool available, using sequential reranking: {num_queries} queries × {num_docs} documents"
        )

        # Get early stopping parameters from config
        min_score = self.config.get("min_score", 0.5)
        target_count = self.config.get("early_stop_target", 5)

        logger.info(
            f"Multi-query reranking: {num_queries} queries, {num_docs} chunks "
            f"(early_stop_target={target_count}, min_score={min_score})"
        )
        for i, query in enumerate(queries, 1):
            query_preview = query[:80] + "..." if len(query) > 80 else query
            logger.info(f"  Query {i}: {query_preview}")

        # Initialize tracking
        score_matrix = [[0.0] * num_queries for _ in range(num_docs)]
        best_scores = [0.0] * num_docs
        docs_passed = set()

        # Rerank query by query with early stopping
        queries_used = 0
        for query_idx, query in enumerate(queries):
            queries_used = query_idx + 1
            logger.info(f"Reranking with query {queries_used}/{num_queries}...")

            reranked = await self.rerank(
                query=query,
                documents=documents,
                top_k=None,
            )

            # Update score matrix and best scores
            for reranked_doc in reranked:
                doc_idx = None

                # Find original document index
                for idx, orig_doc in enumerate(documents):
                    if (orig_doc.get("id") == reranked_doc.get("id")) or (
                        orig_doc.get("text") == reranked_doc.get("text")
                    ):
                        doc_idx = idx
                        break

                if doc_idx is not None:
                    score = reranked_doc.get("rerank_score", 0.0)
                    score_matrix[doc_idx][query_idx] = score

                    # Update best score (max aggregation for early stopping check)
                    if score > best_scores[doc_idx]:
                        best_scores[doc_idx] = score

                    # Track if passed threshold
                    if best_scores[doc_idx] >= min_score:
                        docs_passed.add(doc_idx)

            # Log progress
            logger.info(
                f"After query {queries_used}: {len(docs_passed)} docs passed threshold "
                f"({len(docs_passed)}/{target_count} target)"
            )

            # Check early stopping
            if len(docs_passed) >= target_count:
                saved_queries = num_queries - queries_used
                logger.info(
                    f"Early stopping: {len(docs_passed)} docs passed threshold "
                    f"after {queries_used}/{num_queries} queries (saved {saved_queries} queries)"
                )
                break

        # Aggregate scores
        aggregated_results = []
        for doc_idx, doc in enumerate(documents):
            query_scores = score_matrix[doc_idx][:queries_used]

            if score_aggregation == "max":
                final_score = max(query_scores) if query_scores else 0.0
                best_query_idx = query_scores.index(final_score) if query_scores and final_score > 0 else 0
            elif score_aggregation == "mean":
                final_score = sum(query_scores) / len(query_scores) if query_scores else 0.0
                best_query_idx = query_scores.index(max(query_scores)) if query_scores else 0
            else:
                logger.warning(f"Unknown aggregation: {score_aggregation}, using 'max'")
                final_score = max(query_scores) if query_scores else 0.0
                best_query_idx = query_scores.index(final_score) if query_scores and final_score > 0 else 0

            result_doc = doc.copy()
            result_doc["rerank_score"] = final_score
            result_doc["rerank_best_query_idx"] = best_query_idx
            result_doc["rerank_query_scores"] = {i: score for i, score in enumerate(query_scores)}
            aggregated_results.append(result_doc)

        # Sort by final score (highest first)
        aggregated_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Apply top_k
        if top_k is not None:
            aggregated_results = aggregated_results[:top_k]

        elapsed = time.time() - start_time
        logger.info(f"Multi-query reranking completed in {elapsed:.3f}s using {queries_used}/{num_queries} queries")

        # Log top 10 chunks with score breakdown
        if aggregated_results:
            top_chunks = aggregated_results[:10]
            logger.info(f"Top {len(top_chunks)} chunks with score breakdown:")
            for idx, doc in enumerate(top_chunks, 1):
                chunk_id = doc.get("id", "unknown")
                final_score = doc.get("rerank_score", 0.0)
                best_query_idx = doc.get("rerank_best_query_idx", 0)
                query_scores = doc.get("rerank_query_scores", {})

                score_str = ", ".join([f"Q{i + 1}={s:.3f}" for i, s in query_scores.items()])

                # Get similarity metadata if available
                from ..retrieval.chunk_picking import get_similarity_metadata

                sim_metadata = get_similarity_metadata(chunk_id)
                sim_info = ""
                if sim_metadata:
                    cosine_sim = sim_metadata.get("cosine_similarity", 0.0)
                    sim_rank = sim_metadata.get("similarity_rank", 0)
                    sim_info = f", cosine_sim={cosine_sim:.4f}, sim_rank={sim_rank}"

                content = doc.get("content") or doc.get("text", "")
                content_preview = content[:100] + "..." if len(content) > 100 else content

                logger.info(
                    f"  [{idx}] {chunk_id}: final={final_score:.4f} "
                    f"(best: Q{best_query_idx + 1}, scores: [{score_str}]{sim_info}) | {content_preview}"
                )

        return aggregated_results

    async def close(self):
        """Cleanup resources (mainly for local models)."""
        if self.reranker is not None and hasattr(self.reranker, "__del__"):
            self.reranker.__del__()
        self._initialized = False
        logger.info("RerankProcessor closed")


async def rerank_entities(
    query: str,
    entities: list[dict[str, Any]],
    rerank_config: dict[str, Any],
    rerank_processor: RerankProcessor | None = None,
    stage: str = "stage1",
) -> list[dict[str, Any]]:
    """
    Rerank entities based on their relevance to the query.

    This filters entities by reranking their content/description against the query,
    which reduces the number of candidate chunks and improves performance.

    For Stage 2 entities, uses the best matching shuffled query (highest similarity)
    instead of the original user query for more accurate relevance scoring.

    Args:
        query: The search query (used for Stage 1; Stage 2 uses stage2_best_query)
        entities: List of entity dicts with entity_id, entity_type, description, content
        rerank_config: Entity reranking configuration from config.py
        rerank_processor: RerankProcessor instance (if None, skips reranking)
        stage: "stage1" or "stage2" to control which enable flag to check

    Returns:
        Filtered list of entities that pass the relevance threshold,
        each with added "entity_rerank_score" field
    """
    import time

    # Check if entity reranking is enabled for this stage
    enable_key = f"enable_{stage}"
    if not rerank_config.get(enable_key, False):
        logger.debug(f"Entity reranking disabled for {stage} in config")
        return entities

    if not rerank_processor:
        logger.warning(f"Entity reranking enabled for {stage} but no rerank processor provided, skipping")
        return entities

    if not entities:
        logger.debug("No entities to rerank")
        return entities

    start_time = time.time()
    original_count = len(entities)

    try:
        # Configuration
        min_score = rerank_config.get("min_score", 0.3)
        max_entities = rerank_config.get("max_entities", 200)
        content_field = rerank_config.get("content_field", "content")
        fallback = rerank_config.get("fallback", "use_name")

        # Limit entities to rerank (performance optimization)
        if max_entities > 0 and len(entities) > max_entities:
            logger.info(
                f"Entity reranking: limiting from {len(entities)} to top {max_entities} entities (sorted by degree)"
            )
            entities_to_rerank = entities[:max_entities]
        else:
            entities_to_rerank = entities

        # Prepare documents for reranking
        docs_for_rerank = []
        entity_indices = []  # Track which entities we're reranking
        entity_texts = []  # Track the text used for each entity

        for idx, entity in enumerate(entities_to_rerank):
            entity_id = entity.get("entity_id", "unknown")
            entity_type = entity.get("entity_type", "Unknown")
            description = entity.get("description", "")
            content = entity.get("content", "")

            # Determine text to use for reranking based on content_field setting
            if content_field == "description":
                # Use description field
                text_value = description
            elif content_field == "content":
                # Use content field (full text representation)
                text_value = content
            elif content_field == "both":
                # Concatenate both description and content
                parts = []
                if description:
                    parts.append(description)
                if content:
                    parts.append(content)
                text_value = " | ".join(parts) if parts else ""
            else:
                # Default to content
                text_value = content

            # Build reranking text
            if text_value:
                # Use the selected field value
                text = f"{entity_id} ({entity_type}): {text_value}"
            elif fallback == "use_name":
                # Use entity name only
                text = f"{entity_id} ({entity_type})"
            elif fallback == "keep":
                # No content and fallback is "keep" - skip reranking, assume relevant
                entity["entity_rerank_score"] = 1.0
                continue
            elif fallback == "remove":
                # No content and fallback is "remove" - skip, will be filtered out
                continue
            else:
                # Default: use entity name
                text = f"{entity_id} ({entity_type})"

            docs_for_rerank.append({"text": text})
            entity_indices.append(idx)
            entity_texts.append(text)

        if not docs_for_rerank:
            logger.warning("No entities with valid text for reranking")
            return entities

        logger.info(
            f"Entity reranking ({stage}): reranking {len(docs_for_rerank)} entities "
            f"(min_score={min_score}, content_field={content_field})"
        )

        # For Stage 2, we need to rerank each entity with its best matching shuffled query
        # For Stage 1, use the original user query for all entities
        if stage == "stage2":
            # Stage 2: Rerank each entity individually with its best query
            reranked_docs = []
            for idx, (doc, original_idx, entity_text) in enumerate(zip(docs_for_rerank, entity_indices, entity_texts)):
                entity = entities_to_rerank[original_idx]

                # Get best query for this entity (fallback to original query if not available)
                entity_query = entity.get("stage2_best_query", query)

                # Rerank single entity
                single_reranked = await rerank_processor.rerank(
                    query=entity_query,
                    documents=[doc],
                    top_k=None,
                )

                if single_reranked:
                    reranked_docs.append(single_reranked[0])
                else:
                    # Fallback: assign score 0.0 if reranking failed
                    doc_copy = doc.copy()
                    doc_copy["rerank_score"] = 0.0
                    reranked_docs.append(doc_copy)
        else:
            # Stage 1: Rerank all entities together with original user query
            reranked_docs = await rerank_processor.rerank(
                query=query,
                documents=docs_for_rerank,
                top_k=None,  # Get all scores, filter later
            )

        # Apply scores back to original entities (IN-PLACE) and filter
        filtered_entities = []
        scores_below_threshold = 0

        for doc, original_idx, entity_text in zip(reranked_docs, entity_indices, entity_texts):
            score = doc.get("rerank_score", 0.0)

            # Update original entity IN-PLACE with rerank score
            entities_to_rerank[original_idx]["entity_rerank_score"] = score

            # Filter by threshold
            if score >= min_score:
                filtered_entities.append(entities_to_rerank[original_idx])
            else:
                scores_below_threshold += 1

        elapsed = time.time() - start_time
        not_reranked = original_count - len(entities_to_rerank)

        logger.info(
            f"Entity reranking ({stage}): kept {len(filtered_entities)}/{original_count} entities "
            f"(filtered: {scores_below_threshold} below threshold, {not_reranked} not reranked) "
            f"in {elapsed:.3f}s"
        )

        # Log top entities
        if filtered_entities:
            top_entities = filtered_entities[:5]
            logger.debug("Top reranked entities:")
            for ent in top_entities:
                entity_id = ent.get("entity_id", "unknown")
                score = ent.get("entity_rerank_score", 0.0)
                logger.debug(f"  {entity_id}: score={score:.3f}")

        return filtered_entities

    except Exception as e:
        logger.error(f"CRITICAL: Entity reranking failed: {e}")
        logger.error("Stopping query pipeline due to entity reranking failure")
        raise
