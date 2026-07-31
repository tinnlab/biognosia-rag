"""
Reranking worker pool for parallel cross-encoder scoring.

Creates N persistent workers (default: 10) during app initialization.
Each worker loads the reranker model once and processes batches in parallel.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RerankWorkerPool:
    """
    Pool of persistent reranking workers for parallel scoring.

    Each worker:
    - Loads the reranker model once at startup
    - Waits for (query, documents) tasks
    - Computes rerank scores for the batch
    - Returns (doc_indices, scores)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize reranking worker pool.

        Args:
            config: Reranking configuration with keys:
                - num_workers: Number of parallel workers (default: 10)
                - provider: "local", "jina", "cohere", or "aliyun"
                - model: Model name
                - device: Device for local models ("cuda:0", "cpu")
                - max_length: Max sequence length
                - batch_size: Batch size per worker
        """
        self.config = config or {}
        self.num_workers = self.config.get("num_workers", 10)
        self.provider = self.config.get("provider", "local")

        self.work_queues = []  # One queue per worker
        self.worker_tasks = []
        self._running = False

    async def start(self):
        """Start all reranking workers."""
        if self._running:
            logger.warning("Rerank worker pool already running")
            return

        logger.info(f"Starting rerank worker pool with {self.num_workers} workers...")

        # Create work queues and workers
        for worker_id in range(self.num_workers):
            work_queue = asyncio.Queue()
            self.work_queues.append(work_queue)

            worker_task = asyncio.create_task(self._worker(worker_id, work_queue))
            self.worker_tasks.append(worker_task)

        self._running = True
        logger.info(f"Rerank worker pool started with {self.num_workers} workers")

    async def stop(self):
        """Stop all workers."""
        if not self._running:
            return

        logger.info("Stopping rerank worker pool...")

        # Send stop signals to all workers
        for work_queue in self.work_queues:
            await work_queue.put(None)

        # Wait for all workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        self._running = False
        logger.info("Rerank worker pool stopped")

    async def _worker(self, worker_id: int, work_queue: asyncio.Queue):
        """
        Persistent reranking worker.

        Loads model once at startup, then processes tasks in a loop.
        """
        logger.info(f"[Rerank Worker {worker_id}] Starting and loading model...")

        # Load reranker model (ONCE per worker)
        reranker = None
        if self.provider == "local":
            try:
                from .local_reranker import LocalReranker

                model = self.config.get("model", "BAAI/bge-reranker-v2-m3")
                device = self.config.get("device", "cuda:0")
                max_length = self.config.get("max_length", 512)
                batch_size = self.config.get("batch_size", 32)
                normalize = self.config.get("normalize", True)

                reranker = LocalReranker(
                    model_name=model,
                    device=device,
                    max_length=max_length,
                    batch_size=batch_size,
                    normalize=normalize,
                )
                reranker.initialize()
                logger.info(f"[Rerank Worker {worker_id}] Model loaded on {device}")
            except Exception as e:
                logger.error(f"[Rerank Worker {worker_id}] Failed to load model: {e}")
                return
        else:
            logger.error(f"[Rerank Worker {worker_id}] Non-local providers not supported in worker pool")
            return

        logger.info(f"[Rerank Worker {worker_id}] Ready and waiting for tasks...")

        # Process tasks
        while True:
            try:
                # Wait for task
                task = await work_queue.get()

                # Check for stop signal
                if task is None:
                    logger.info(f"[Rerank Worker {worker_id}] Received stop signal")
                    break

                # Unpack task
                task_id, query, documents, result_queue = task

                logger.debug(
                    f"[Rerank Worker {worker_id}] Processing task {task_id}: "
                    f"query={query[:50]}..., docs={len(documents)}"
                )

                # Extract texts
                texts = [doc.get("text", "") for doc in documents]

                # Compute scores
                indexed_scores = await reranker.rerank(query, texts, top_k=None)

                # Send results
                await result_queue.put((task_id, indexed_scores))

                logger.debug(f"[Rerank Worker {worker_id}] Task {task_id} complete: {len(indexed_scores)} scores")

            except Exception as e:
                logger.error(f"[Rerank Worker {worker_id}] Error processing task: {e}", exc_info=True)
                if "result_queue" in locals():
                    await result_queue.put((task_id, []))

        logger.info(f"[Rerank Worker {worker_id}] Shutting down")

    async def rerank_all_pairs(
        self,
        queries: list[str],
        documents: list[dict[str, Any]],
        score_aggregation: str = "max",
    ) -> list[dict[str, Any]]:
        """
        Rerank all query-document pairs in parallel across workers.

        No early stopping - computes scores for ALL query×document pairs,
        then aggregates using max (or mean) across queries.

        Args:
            queries: List of query strings
            documents: List of document dicts with "text" field
            score_aggregation: "max" or "mean"

        Returns:
            List of documents with rerank scores, sorted by score
        """
        import time
        import uuid

        if not queries or not documents:
            return []

        if not self._running:
            raise RuntimeError("Rerank worker pool not started. Call await start() first.")

        start_time = time.time()
        num_queries = len(queries)
        num_docs = len(documents)

        logger.info(
            f"Parallel reranking: {num_queries} queries × {num_docs} documents = "
            f"{num_queries * num_docs} pairs across {self.num_workers} workers"
        )

        # Initialize score matrix
        score_matrix = [[0.0] * num_queries for _ in range(num_docs)]

        # Create result queue for collecting results
        result_queue = asyncio.Queue()

        # Distribute work across workers
        # Strategy: Assign each query to a worker in round-robin fashion
        tasks_submitted = 0
        for query_idx, query in enumerate(queries):
            worker_id = query_idx % self.num_workers
            task_id = f"{uuid.uuid4().hex[:8]}-q{query_idx}"

            await self.work_queues[worker_id].put((task_id, query, documents, result_queue))
            tasks_submitted += 1

        logger.info(f"Submitted {tasks_submitted} tasks to {self.num_workers} workers")

        # Collect results
        for _ in range(tasks_submitted):
            task_id, indexed_scores = await result_queue.get()

            # Extract query index from task_id
            query_idx = int(task_id.split("-q")[1])

            # Update score matrix
            for doc_idx, score in indexed_scores:
                score_matrix[doc_idx][query_idx] = score

        logger.info("All workers completed, aggregating scores...")

        # Aggregate scores
        aggregated_results = []
        for doc_idx, doc in enumerate(documents):
            query_scores = score_matrix[doc_idx]

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

        elapsed = time.time() - start_time
        logger.info(
            f"Parallel reranking completed in {elapsed:.3f}s "
            f"({num_queries * num_docs} pairs, {elapsed / (num_queries * num_docs) * 1000:.2f}ms per pair)"
        )

        # Log top 10 chunks with score breakdown
        if aggregated_results:
            top_chunks = aggregated_results[:10]
            logger.info(f"Top {len(top_chunks)} chunks after parallel reranking:")
            for idx, doc in enumerate(top_chunks, 1):
                chunk_id = doc.get("id", "unknown")
                final_score = doc.get("rerank_score", 0.0)
                best_query_idx = doc.get("rerank_best_query_idx", 0)
                query_scores = doc.get("rerank_query_scores", {})

                score_str = ", ".join([f"Q{i + 1}={s:.3f}" for i, s in query_scores.items()])

                content = doc.get("content") or doc.get("text", "")
                content_preview = content[:100] + "..." if len(content) > 100 else content

                logger.info(
                    f"  [{idx}] {chunk_id}: final={final_score:.4f} "
                    f"(best: Q{best_query_idx + 1}, scores: [{score_str}]) | {content_preview}"
                )

        return aggregated_results
