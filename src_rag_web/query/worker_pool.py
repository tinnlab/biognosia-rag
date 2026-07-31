"""
Worker pool for parallel retrieval tasks.

Creates 3 persistent worker tasks during initialization:
1. KG retrieval worker (entities + relationships)
2. Elasticsearch retrieval worker (keyword expansion + BM25)
3. Milvus retrieval worker (query expansion + semantic search)

Workers stay alive and are reused for all queries.
Uses asyncio for concurrent I/O operations.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RetrievalWorkerPool:
    """
    Pool of 3 persistent workers for concurrent retrieval.

    Workers are asyncio tasks created during initialization and reused for all queries.
    Optimized for I/O-bound operations (database queries, HTTP requests).
    """

    def __init__(self):
        self.kg_queue = asyncio.Queue()
        self.es_queue = asyncio.Queue()
        self.milvus_queue = asyncio.Queue()

        self.kg_worker_task = None
        self.es_worker_task = None
        self.milvus_worker_task = None

        self._running = False

    async def start(self):
        """Start the 3 persistent worker tasks."""
        if self._running:
            logger.warning("Worker pool already running")
            return

        logger.info("Starting worker pool with 3 persistent workers (asyncio tasks)...")

        self.kg_worker_task = asyncio.create_task(self._kg_worker())
        self.es_worker_task = asyncio.create_task(self._es_worker())
        self.milvus_worker_task = asyncio.create_task(self._milvus_worker())

        self._running = True
        logger.info("Worker pool started successfully")

    async def stop(self):
        """Stop all worker tasks."""
        if not self._running:
            return

        logger.info("Stopping worker pool...")

        # Send stop signals
        await self.kg_queue.put(None)
        await self.es_queue.put(None)
        await self.milvus_queue.put(None)

        # Wait for workers to finish
        if self.kg_worker_task:
            await self.kg_worker_task
        if self.es_worker_task:
            await self.es_worker_task
        if self.milvus_worker_task:
            await self.milvus_worker_task

        self._running = False
        logger.info("Worker pool stopped")

    async def _kg_worker(self):
        """
        Persistent KG retrieval worker.

        Waits for tasks on kg_queue and processes them.
        """
        logger.info("[KG Worker] Started and waiting for tasks...")

        while True:
            try:
                # Wait for task
                task = await self.kg_queue.get()

                # Check for stop signal
                if task is None:
                    logger.info("[KG Worker] Received stop signal, shutting down")
                    break

                # Unpack task
                (
                    task_id,
                    query,
                    entity_info,
                    relationship_info,
                    text_chunks_storage,
                    chunk_entity_relation_storage,
                    chunks_vdb,
                    embedding_func,
                    param,
                    llm_provider,
                    result_queue,
                    detailed_logger,
                ) = task

                logger.info(f"[KG Worker] Processing task {task_id}")

                # Execute KG retrieval
                from ..retrieval.kg_search import (
                    find_related_text_unit_from_entities,
                    find_related_text_unit_from_relations,
                )

                # Get entity chunk IDs
                entity_chunk_ids, _ = await find_related_text_unit_from_entities(
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
                    enable_candidate_filtering=False,
                    detailed_logger=detailed_logger,
                )

                # Get relation chunk IDs
                relation_chunk_ids, _ = await find_related_text_unit_from_relations(
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
                    enable_candidate_filtering=False,
                    detailed_logger=detailed_logger,
                )

                logger.info(
                    f"[KG Worker] Task {task_id} complete: "
                    f"{len(entity_chunk_ids)} entity chunks, {len(relation_chunk_ids)} relation chunks"
                )

                # Send result
                await result_queue.put((entity_chunk_ids, relation_chunk_ids))

            except Exception as e:
                logger.error(f"[KG Worker] Error processing task: {e}", exc_info=True)
                # Put exception in result queue to propagate to caller
                if "result_queue" in locals():
                    await result_queue.put(("ERROR", e))

    async def _es_worker(self):
        """
        Persistent Elasticsearch retrieval worker.

        Waits for tasks on es_queue and processes them.
        """
        logger.info("[ES Worker] Started and waiting for tasks...")

        while True:
            try:
                # Wait for task
                task = await self.es_queue.get()

                # Check for stop signal
                if task is None:
                    logger.info("[ES Worker] Received stop signal, shutting down")
                    break

                # Unpack task
                (
                    task_id,
                    query,
                    es_client,
                    llm_provider,
                    config,
                    cache_manager,
                    result_queue,
                    detailed_logger,
                    param,
                    query_expansions,  # Pre-generated
                    hyde_answers,  # Pre-generated
                ) = task

                logger.info(f"[ES Worker] Processing task {task_id}")

                # Execute ES retrieval
                from ..retrieval.elasticsearch_search import search_elasticsearch

                hybrid_config = config.get("hybrid_search", {})
                use_expansions = hybrid_config.get("use_query_expansions_for_es", True)

                if use_expansions:
                    # Use pre-generated query expansions + HyDE for ES search
                    all_queries = query_expansions + hyde_answers
                    logger.info(
                        f"[ES Worker] Task {task_id} using multi-query search: {len(all_queries)} queries "
                        f"({len(query_expansions)} expansions + {len(hyde_answers)} HyDE)"
                    )

                    # Search ES with each query in PARALLEL (like Milvus does)
                    all_chunk_results = {}  # chunk_id -> (score, result)
                    es_top_k_per_query = hybrid_config.get("es_top_k_per_query", 5000)

                    # Define async search function
                    async def search_single_es_query(q_text):
                        return await search_elasticsearch(
                            keywords=q_text,
                            es_client=es_client,
                            top_k=es_top_k_per_query,
                            detailed_logger=None,
                        )

                    # Execute all ES searches in parallel
                    import asyncio as async_module

                    search_tasks = [search_single_es_query(q) for q in all_queries]
                    all_es_results = await async_module.gather(*search_tasks)

                    logger.info(f"[ES Worker] Completed {len(all_queries)} parallel ES searches")

                    # Aggregate results using MAX score
                    for q_results in all_es_results:
                        for r in q_results:
                            chunk_id = r["chunk_id"]
                            score = r["score"]
                            if chunk_id not in all_chunk_results or score > all_chunk_results[chunk_id][0]:
                                all_chunk_results[chunk_id] = (score, r)

                    # Sort by score and limit to es_top_k
                    sorted_results = sorted(all_chunk_results.items(), key=lambda x: x[1][0], reverse=True)
                    es_top_k_final = hybrid_config.get("es_top_k", 50000)
                    top_results = sorted_results[:es_top_k_final]

                    es_results = [result for _, (score, result) in top_results]
                    chunk_ids = [r["chunk_id"] for r in es_results]

                    logger.info(
                        f"[ES Worker] Task {task_id} multi-query complete: {len(chunk_ids)} unique chunks "
                        f"from {len(all_queries)} queries"
                    )

                    # Log to detailed logger
                    if detailed_logger:
                        for r in es_results:
                            detailed_logger.log_retrieval_elasticsearch_chunk(r)

                    # Send result with metadata (same format as Milvus)
                    await result_queue.put(
                        (
                            chunk_ids,
                            {
                                "all_queries": all_queries,
                                "expansions": query_expansions,
                                "hyde": hyde_answers,
                            },
                        )
                    )

                else:
                    # Original keyword expansion method
                    from ..retrieval.keyword_expansion import expand_query_keywords

                    keyword_data = await expand_query_keywords(
                        query=query,
                        llm_provider=llm_provider,
                        config=config,
                        cache_manager=cache_manager,
                    )
                    keywords = keyword_data["keyword_string"]

                    logger.info(
                        f"[ES Worker] Task {task_id} keyword expansion: "
                        f"{len(keyword_data.get('all_keywords', []))} keywords"
                    )

                    es_results = await search_elasticsearch(
                        keywords=keywords,
                        es_client=es_client,
                        top_k=hybrid_config.get("es_top_k", 50),
                        detailed_logger=detailed_logger,
                    )

                    chunk_ids = [r["chunk_id"] for r in es_results]
                    logger.info(f"[ES Worker] Task {task_id} complete: {len(chunk_ids)} chunks")

                    await result_queue.put((chunk_ids, keyword_data.get("all_keywords", [])))

            except Exception as e:
                logger.error(f"[ES Worker] Error processing task: {e}", exc_info=True)
                # Put exception in result queue to propagate to caller
                if "result_queue" in locals():
                    await result_queue.put(("ERROR", e))

    async def _milvus_worker(self):
        """
        Persistent Milvus retrieval worker.

        Waits for tasks on milvus_queue and processes them.
        """
        logger.info("[Milvus Worker] Started and waiting for tasks...")

        while True:
            try:
                # Wait for task
                task = await self.milvus_queue.get()

                # Check for stop signal
                if task is None:
                    logger.info("[Milvus Worker] Received stop signal, shutting down")
                    break

                # Unpack task
                (
                    task_id,
                    query,
                    embedding_manager,
                    chunks_vdb,
                    llm_provider,
                    config,
                    param,
                    result_queue,
                    detailed_logger,
                    query_expansions,  # Pre-generated
                    hyde_answers,  # Pre-generated
                ) = task

                logger.info(f"[Milvus Worker] Processing task {task_id}")

                # Execute Milvus retrieval
                from ..retrieval.chunk_picking import get_candidate_chunks_from_vector_search

                # Use pre-generated expansions + HyDE
                all_queries = query_expansions + hyde_answers
                logger.info(
                    f"[Milvus Worker] Task {task_id} using {len(all_queries)} queries "
                    f"({len(query_expansions)} expansions + {len(hyde_answers)} HyDE)"
                )

                # Milvus vector search
                hybrid_config = config.get("hybrid_search", {})
                milvus_top_k = hybrid_config.get("milvus_top_k", 50)

                chunk_ids = await get_candidate_chunks_from_vector_search(
                    queries=all_queries,
                    chunks_vdb=chunks_vdb,
                    embedding_func=embedding_manager.embed_chunks,
                    top_k_per_query=param.candidate_top_k,
                    final_top_k=milvus_top_k,
                    detailed_logger=detailed_logger,
                )

                logger.info(f"[Milvus Worker] Task {task_id} complete: {len(chunk_ids)} chunks")

                # Send result with metadata
                await result_queue.put(
                    (
                        chunk_ids,
                        {
                            "all_queries": all_queries,
                            "expansions": query_expansions,
                            "hyde": hyde_answers,
                        },
                    )
                )

            except Exception as e:
                logger.error(f"[Milvus Worker] Error processing task: {e}", exc_info=True)
                # Put exception in result queue to propagate to caller
                if "result_queue" in locals():
                    await result_queue.put(("ERROR", e))

    async def retrieve_parallel(
        self,
        query: str,
        entity_info: list,
        relationship_info: list,
        text_chunks_storage: Any,
        chunk_entity_relation_storage: Any,
        chunks_vdb: Any,
        embedding_manager: Any,
        param: Any,
        llm_provider: Any,
        es_client: Any,
        config: dict,
        cache_manager: Any,
        use_hybrid: bool,
        detailed_logger=None,
    ) -> tuple:
        """
        Submit retrieval tasks to the 3 persistent workers and wait for results.

        Returns:
            Tuple of ((entity_chunk_ids, relation_chunk_ids),
                      (es_chunk_ids, expanded_keywords),
                      (milvus_chunk_ids, query_expansions))
        """
        import uuid

        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[Worker Pool] Submitting task {task_id} to 3 workers...")

        # Generate query expansions + HyDE + decomposition ONCE (in parallel)
        # - Expansions + HyDE: for Milvus/ES retrieval (vocabulary diversity)
        # - Decomposition: for reranking (structural breakdown of complex questions)
        query_expansions = []
        hyde_answers = []
        decomposed_queries = []

        # Only generate query expansions/HyDE if vector or keyword retrieval enabled
        if param.enable_vector_retrieval or param.enable_keyword_retrieval:
            # Run expansion, HyDE, and decomposition in parallel
            from ..retrieval.query_decomposition import decompose_query
            from ..retrieval.query_expansion import expand_query_for_retrieval, generate_hyde

            async def run_expansion():
                # Only generate if num_query_expansions > 0
                if param.num_query_expansions > 0:
                    try:
                        result = await expand_query_for_retrieval(
                            query=query,
                            llm_provider=llm_provider,
                            num_expansions=param.num_query_expansions,
                            min_expansions=param.min_query_expansions,
                            max_expansions=param.max_query_expansions,
                            enable=True,
                            max_tokens=param.max_tokens_query_expansion,
                            context="for Milvus/ES search",
                        )
                        logger.info(f"[Worker Pool] Generated {len(result)} query expansions")
                        return result
                    except Exception as e:
                        logger.error(f"[Worker Pool] Query expansion failed: {e}")
                        return [query]
                return [query]  # Return original query if expansions disabled

            async def run_hyde():
                if param.enable_hyde and (param.enable_vector_retrieval or param.enable_keyword_retrieval):
                    try:
                        result = await generate_hyde(
                            query,
                            llm_provider,
                            min_hyde=param.min_hyde_expansions,
                            max_hyde=param.max_hyde_expansions,
                            max_tokens=param.max_tokens_hyde,
                            temperature=param.hyde_temperature,
                        )
                        logger.info(f"[Worker Pool] Generated {len(result)} HyDE answers")
                        return result
                    except Exception as e:
                        logger.error(f"[Worker Pool] HyDE generation failed: {e}")
                        return []
                return []

            async def run_decomposition():
                if param.enable_query_decomposition:
                    try:
                        result = await decompose_query(
                            query,
                            llm_provider,
                            max_queries=param.max_decomposed_queries,
                            enable=True,
                            max_tokens=param.max_tokens_decomposition,
                            temperature=param.decomposition_temperature,
                        )
                        logger.info(f"[Worker Pool] Generated {len(result)} decomposed queries")
                        return result
                    except Exception as e:
                        logger.error(f"[Worker Pool] Query decomposition failed: {e}")
                        return [query]
                return []

            # Run all three in parallel
            query_expansions, hyde_answers, decomposed_queries = await asyncio.gather(
                run_expansion(),
                run_hyde(),
                run_decomposition(),
            )

        # Create result queues
        kg_result_queue = asyncio.Queue()
        es_result_queue = asyncio.Queue()
        milvus_result_queue = asyncio.Queue()

        # Track which workers to wait for
        workers_to_wait = []

        # Submit tasks to workers (conditionally based on component flags)
        if param.enable_entity_retrieval:
            logger.info("[Worker Pool] Submitting to KG worker (entity retrieval enabled)")
            await self.kg_queue.put(
                (
                    task_id,
                    query,
                    entity_info,
                    relationship_info,
                    text_chunks_storage,
                    chunk_entity_relation_storage,
                    chunks_vdb,
                    embedding_manager.embed_chunks,
                    param,
                    llm_provider,
                    kg_result_queue,
                    detailed_logger,
                )
            )
            workers_to_wait.append(("kg", kg_result_queue.get()))
        else:
            logger.info("[Worker Pool] Skipping KG worker (entity retrieval disabled)")

        if param.enable_vector_retrieval:
            logger.info("[Worker Pool] Submitting to Milvus worker (vector retrieval enabled)")
            await self.milvus_queue.put(
                (
                    task_id,
                    query,
                    embedding_manager,
                    chunks_vdb,
                    llm_provider,
                    config,
                    param,
                    milvus_result_queue,
                    detailed_logger,
                    query_expansions,  # Pre-generated
                    hyde_answers,  # Pre-generated
                )
            )
            workers_to_wait.append(("milvus", milvus_result_queue.get()))
        else:
            logger.info("[Worker Pool] Skipping Milvus worker (vector retrieval disabled)")

        if use_hybrid and param.enable_keyword_retrieval:
            logger.info("[Worker Pool] Submitting to ES worker (keyword retrieval enabled)")
            await self.es_queue.put(
                (
                    task_id,
                    query,
                    es_client,
                    llm_provider,
                    config,
                    cache_manager,
                    es_result_queue,
                    detailed_logger,
                    param,
                    query_expansions,  # Pre-generated
                    hyde_answers,  # Pre-generated
                )
            )
            workers_to_wait.append(("es", es_result_queue.get()))
        else:
            if not use_hybrid:
                logger.info("[Worker Pool] Skipping ES worker (hybrid search disabled)")
            elif not param.enable_keyword_retrieval:
                logger.info("[Worker Pool] Skipping ES worker (keyword retrieval disabled)")

        # Wait for results from submitted workers
        logger.info(f"[Worker Pool] Waiting for results from {len(workers_to_wait)} workers...")

        # Gather results only from submitted workers
        if workers_to_wait:
            worker_names = [name for name, _ in workers_to_wait]
            worker_futures = [future for _, future in workers_to_wait]
            results = await asyncio.gather(*worker_futures)

            # Map results back to their worker types
            result_map = dict(zip(worker_names, results))
            kg_result = result_map.get("kg", ([], []))
            milvus_result = result_map.get("milvus", ([], []))
            es_result = result_map.get("es", ([], []))
        else:
            # No workers submitted (all disabled) - return empty results
            logger.warning("[Worker Pool] No workers submitted - all components disabled!")
            kg_result = ([], [])
            milvus_result = ([], [])
            es_result = ([], [])

        # Check for errors from workers and re-raise (only for enabled workers)
        if kg_result and len(kg_result) > 0 and kg_result[0] == "ERROR":
            logger.error("[Worker Pool] KG worker failed, propagating exception")
            raise kg_result[1]
        if es_result and len(es_result) > 0 and es_result[0] == "ERROR":
            logger.error("[Worker Pool] ES worker failed, propagating exception")
            raise es_result[1]
        if milvus_result and len(milvus_result) > 0 and milvus_result[0] == "ERROR":
            logger.error("[Worker Pool] Milvus worker failed, propagating exception")
            raise milvus_result[1]

        enabled_count = sum(
            [
                param.enable_entity_retrieval,
                param.enable_vector_retrieval,
                param.enable_keyword_retrieval and use_hybrid,
            ]
        )
        logger.info(f"[Worker Pool] Task {task_id} completed by {enabled_count} enabled workers")

        return kg_result, es_result, milvus_result, decomposed_queries
