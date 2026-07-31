"""
Two-stage reranking worker pool for cascade filtering.

Stage 1: Fast model (TinyBERT) filters to top-K candidates
Stage 2: Precise model (from main config) ranks final results

Architecture:
- Multiprocessing-based: Each worker is a separate process with independent CUDA context
- Document-level parallelism: Documents are split across all workers
- All workers process different document slices in true parallel

This provides 3-5x speedup from cascade + true parallel GPU execution.
"""

import logging
import multiprocessing as mp
import time
from typing import Any

logger = logging.getLogger(__name__)


def _stage1_worker_process(worker_id: int, task_queue: mp.Queue, result_queue: mp.Queue, config: dict):
    """
    Stage 1 worker process function (runs in separate process).

    Each process loads its own model copy and processes tasks from queue.
    """

    # Set process title for easier identification
    try:
        import setproctitle

        setproctitle.setproctitle(f"rerank-s1-worker-{worker_id}")
    except ImportError:
        pass

    logger.info(f"[Stage 1 Worker {worker_id}] Process starting (PID: {mp.current_process().pid})...")

    # Load stage 1 reranker (fast model) - each process has its own copy
    reranker = None
    try:
        from .local_reranker import LocalReranker

        model = config.get("model", "cross-encoder/ms-marco-TinyBERT-L2-v2")
        device = config.get("device", "cuda:1")
        max_length = config.get("max_length", 512)
        batch_size = config.get("batch_size", 4000)

        reranker = LocalReranker(
            model_name=model,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            normalize=True,
        )
        reranker.initialize()
        logger.info(f"[Stage 1 Worker {worker_id}] Model {model} loaded on {device} (PID: {mp.current_process().pid})")
    except Exception as e:
        logger.error(f"[Stage 1 Worker {worker_id}] Failed to load model: {e}")
        return

    logger.info(f"[Stage 1 Worker {worker_id}] Ready and waiting for tasks...")

    # Process tasks from queue
    while True:
        try:
            task = task_queue.get()

            if task is None:
                logger.info(f"[Stage 1 Worker {worker_id}] Received stop signal")
                break

            task_id, query, documents = task

            logger.debug(
                f"[Stage 1 Worker {worker_id}] Processing task {task_id}: query={query[:50]}..., docs={len(documents)}"
            )

            # Extract texts
            # Extract texts (try "text" then "content")
            texts = [doc.get("text") or doc.get("content", "") for doc in documents]

            # Compute scores synchronously (no async needed in subprocess)
            indexed_scores = []
            if texts:
                scores = reranker.compute_scores(query, texts)
                indexed_scores = list(enumerate(scores))
                # Sort by score descending
                indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Send results back
            result_queue.put((task_id, indexed_scores))

            logger.debug(f"[Stage 1 Worker {worker_id}] Task {task_id} complete: {len(indexed_scores)} scores")

        except Exception as e:
            logger.error(f"[Stage 1 Worker {worker_id}] Error processing task: {e}", exc_info=True)
            if "task_id" in locals():
                result_queue.put((task_id, []))

    logger.info(f"[Stage 1 Worker {worker_id}] Shutting down")


def _stage2_worker_process(worker_id: int, task_queue: mp.Queue, result_queue: mp.Queue, config: dict):
    """
    Stage 2 worker process function (runs in separate process).

    Each process loads its own model copy and processes tasks from queue.
    """

    # Set process title for easier identification
    try:
        import setproctitle

        setproctitle.setproctitle(f"rerank-s2-worker-{worker_id}")
    except ImportError:
        pass

    logger.info(f"[Stage 2 Worker {worker_id}] Process starting (PID: {mp.current_process().pid})...")

    # Load stage 2 reranker (precise model) - each process has its own copy
    reranker = None
    try:
        from .local_reranker import LocalReranker

        model = config.get("model", "jinaai/jina-reranker-v2-base-multilingual")
        device = config.get("device", "cuda:1")
        max_length = config.get("max_length", 1024)
        batch_size = config.get("batch_size", 2000)

        reranker = LocalReranker(
            model_name=model,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            normalize=True,
        )
        reranker.initialize()
        logger.info(f"[Stage 2 Worker {worker_id}] Model {model} loaded on {device} (PID: {mp.current_process().pid})")
    except Exception as e:
        logger.error(f"[Stage 2 Worker {worker_id}] Failed to load model: {e}")
        return

    logger.info(f"[Stage 2 Worker {worker_id}] Ready and waiting for tasks...")

    # Process tasks from queue
    while True:
        try:
            task = task_queue.get()

            if task is None:
                logger.info(f"[Stage 2 Worker {worker_id}] Received stop signal")
                break

            task_id, query, documents = task

            logger.debug(
                f"[Stage 2 Worker {worker_id}] Processing task {task_id}: query={query[:50]}..., docs={len(documents)}"
            )

            # Extract texts
            # Extract texts (try "text" then "content")
            texts = [doc.get("text") or doc.get("content", "") for doc in documents]

            # Compute scores synchronously (no async needed in subprocess)
            indexed_scores = []
            if texts:
                scores = reranker.compute_scores(query, texts)
                indexed_scores = list(enumerate(scores))
                # Sort by score descending
                indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Send results back
            result_queue.put((task_id, indexed_scores))

            logger.debug(f"[Stage 2 Worker {worker_id}] Task {task_id} complete: {len(indexed_scores)} scores")

        except Exception as e:
            logger.error(f"[Stage 2 Worker {worker_id}] Error processing task: {e}", exc_info=True)
            if "task_id" in locals():
                result_queue.put((task_id, []))

    logger.info(f"[Stage 2 Worker {worker_id}] Shutting down")


class TwoStageRerankerPool:
    """
    Two-stage cascade reranking pool with multiprocessing workers.

    Architecture:
    - Stage 1 pool: N separate processes with fast TinyBERT model
    - Stage 2 pool: M separate processes with precise model (from main config)
    - Each process has independent CUDA context for true parallel GPU execution
    - Document-level parallelism: Documents split across all workers

    Workflow:
    1. Split documents into N slices (one per stage 1 worker)
    2. All N processes execute GPU work in true parallel
    3. Aggregate stage 1 results, filter to top-K candidates globally
    4. Split top-K candidates into M slices (one per stage 2 worker)
    5. All M processes execute GPU work in true parallel
    6. Aggregate stage 2 results, produce final rankings
    7. Repeat for remaining queries
    """

    def __init__(
        self,
        stage1_config: dict[str, Any],
        stage2_config: dict[str, Any],
        two_stage_config: dict[str, Any],
    ):
        """
        Initialize two-stage reranker pool.

        Args:
            stage1_config: Stage 1 reranker config (fast model)
                - model: Model name (default: cross-encoder/ms-marco-TinyBERT-L2-v2)
                - device: Device (default: cuda:1)
                - max_length: Max sequence length (default: 512)
                - batch_size: Batch size (default: 4000)
                - num_workers: Number of workers (default: 10)
            stage2_config: Stage 2 reranker config (precise model)
                - Uses settings from main [rerank] section
            two_stage_config: Two-stage settings
                - enabled: Enable two-stage reranking
                - stage1_top_k: Number of candidates to keep after stage 1
        """
        self.stage1_config = stage1_config
        self.stage2_config = stage2_config
        self.two_stage_config = two_stage_config

        self.stage1_top_k = two_stage_config.get("stage1_top_k", 10000)
        self.stage1_min_score = two_stage_config.get("stage1_min_score", 0.0)
        self.stage2_min_score = two_stage_config.get("stage2_min_score", 0.5)
        self.stage1_num_workers = stage1_config.get("num_workers", 10)
        self.stage2_num_workers = stage2_config.get("num_workers", 7)

        # Multiprocessing queues and processes
        self.stage1_task_queue = None
        self.stage1_result_queue = None
        self.stage1_processes = []

        self.stage2_task_queue = None
        self.stage2_result_queue = None
        self.stage2_processes = []

        self._running = False

    async def start(self):
        """Start both stage 1 and stage 2 worker processes."""
        if self._running:
            logger.warning("Two-stage reranker pool already running")
            return

        logger.info(
            f"Starting two-stage reranker pool with multiprocessing "
            f"(Stage 1: {self.stage1_num_workers} processes, Stage 2: {self.stage2_num_workers} processes)..."
        )

        # CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
        # 'fork' doesn't work with CUDA - causes "Cannot re-initialize CUDA in forked subprocess"
        try:
            mp.set_start_method("spawn", force=True)
            logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
        except RuntimeError:
            # Start method already set, check if it's spawn
            if mp.get_start_method() != "spawn":
                logger.warning(
                    f"Multiprocessing start method is '{mp.get_start_method()}', not 'spawn'. "
                    f"This may cause CUDA errors. Consider setting it to 'spawn' at application startup."
                )

        # Create multiprocessing context with spawn method
        ctx = mp.get_context("spawn")

        # Create multiprocessing queues using spawn context
        self.stage1_task_queue = ctx.Queue()
        self.stage1_result_queue = ctx.Queue()
        self.stage2_task_queue = ctx.Queue()
        self.stage2_result_queue = ctx.Queue()

        # Start stage 1 worker processes using spawn context
        logger.info("Starting Stage 1 worker processes (fast filtering)...")
        for worker_id in range(self.stage1_num_workers):
            process = ctx.Process(
                target=_stage1_worker_process,
                args=(worker_id, self.stage1_task_queue, self.stage1_result_queue, self.stage1_config),
                daemon=False,  # Non-daemon so we can control shutdown
            )
            process.start()
            self.stage1_processes.append(process)
            logger.info(f"Started Stage 1 Worker {worker_id} (PID: {process.pid})")

        # Start stage 2 worker processes using spawn context
        logger.info("Starting Stage 2 worker processes (precise ranking)...")
        for worker_id in range(self.stage2_num_workers):
            process = ctx.Process(
                target=_stage2_worker_process,
                args=(worker_id, self.stage2_task_queue, self.stage2_result_queue, self.stage2_config),
                daemon=False,  # Non-daemon so we can control shutdown
            )
            process.start()
            self.stage2_processes.append(process)
            logger.info(f"Started Stage 2 Worker {worker_id} (PID: {process.pid})")

        self._running = True

        # Give workers time to load models (spawn is slower than fork)
        # Each worker needs to initialize Python interpreter + load transformer models
        import asyncio

        logger.info("Waiting for workers to load models...")
        await asyncio.sleep(5)

        logger.info(
            f"Two-stage reranker pool started (Stage 1: {self.stage1_num_workers} processes, "
            f"Stage 2: {self.stage2_num_workers} processes)"
        )

    async def stop(self):
        """Stop both worker process pools."""
        if not self._running:
            return

        logger.info("Stopping two-stage reranker pool...")

        # Send stop signals to all stage 1 workers
        for _ in range(self.stage1_num_workers):
            self.stage1_task_queue.put(None)

        # Send stop signals to all stage 2 workers
        for _ in range(self.stage2_num_workers):
            self.stage2_task_queue.put(None)

        # Wait for all processes to finish (with timeout)
        timeout = 10.0
        for process in self.stage1_processes + self.stage2_processes:
            process.join(timeout=timeout)
            if process.is_alive():
                logger.warning(f"Process {process.pid} did not terminate gracefully, forcing...")
                process.terminate()
                process.join(timeout=2)

        # Clean up
        self.stage1_processes.clear()
        self.stage2_processes.clear()

        self._running = False
        logger.info("Two-stage reranker pool stopped")

    async def rerank_two_stage(
        self,
        queries: list[str],
        documents: list[dict[str, Any]],
        score_aggregation: str = "max",
        detailed_logger=None,
    ) -> list[dict[str, Any]]:
        """
        Two-stage cascade reranking with multiprocessing.

        Stage 1: Fast model filters to top-K candidates
        Stage 2: Precise model ranks final results

        Args:
            queries: List of query strings
            documents: List of document dicts with "text" field
            score_aggregation: "max" or "mean" (for multi-query aggregation)
            detailed_logger: Optional DetailedLogger instance for structured logging

        Returns:
            List of documents with final rerank scores, sorted by score
        """
        import uuid

        if not queries or not documents:
            return []

        if not self._running:
            raise RuntimeError("Two-stage reranker pool not started. Call await start() first.")

        overall_start = time.time()
        num_queries = len(queries)
        num_docs = len(documents)

        logger.info(
            f"=== TWO-STAGE RERANKING START ==="
            f"\n  Queries: {num_queries}"
            f"\n  Documents: {num_docs}"
            f"\n  Stage 1 filter: top {self.stage1_top_k}"
            f"\n  Score aggregation: {score_aggregation}"
        )

        # OPTIMIZATION: Skip Stage 1 if we have fewer docs than stage1_top_k
        if num_docs <= self.stage1_top_k:
            logger.info(
                f"Skipping Stage 1: only {num_docs} documents (<= {self.stage1_top_k}). "
                f"Going directly to Stage 2 (precise model)."
            )
            top_k_documents = documents
            stage1_elapsed = 0.0
        else:
            # ===== STAGE 1: Fast filtering to top-K =====
            stage1_start = time.time()
            logger.info(f"[Stage 1] Filtering {num_docs} documents to top {self.stage1_top_k}...")

            # Initialize score matrix
            score_matrix_stage1 = [[0.0] * num_queries for _ in range(num_docs)]

            # Split documents across workers for parallel processing
            docs_per_worker = (num_docs + self.stage1_num_workers - 1) // self.stage1_num_workers

            # Submit tasks: split documents across workers for each query
            tasks_submitted = 0
            task_map = {}  # task_id -> (query_idx, worker_id)

            for query_idx, query in enumerate(queries):
                for worker_id in range(self.stage1_num_workers):
                    # Calculate document slice for this worker
                    start_idx = worker_id * docs_per_worker
                    end_idx = min(start_idx + docs_per_worker, num_docs)

                    if start_idx >= num_docs:
                        break  # No more documents for this worker

                    doc_slice = documents[start_idx:end_idx]
                    task_id = f"{uuid.uuid4().hex[:8]}-s1-q{query_idx}-w{worker_id}"
                    task_map[task_id] = (query_idx, worker_id)

                    self.stage1_task_queue.put((task_id, query, doc_slice))
                    tasks_submitted += 1

            logger.info(
                f"[Stage 1] Submitted {tasks_submitted} tasks to {self.stage1_num_workers} processes "
                f"({num_queries} queries × ~{self.stage1_num_workers} workers, {docs_per_worker} docs/worker)"
            )

            # Collect stage 1 results
            for _ in range(tasks_submitted):
                task_id, indexed_scores = self.stage1_result_queue.get()

                # Get query index and worker ID
                query_idx, worker_id = task_map[task_id]

                # Calculate document offset for this worker's slice
                doc_offset = worker_id * docs_per_worker

                # Store scores in matrix (adjust indices for document slice)
                for local_doc_idx, score in indexed_scores:
                    global_doc_idx = doc_offset + local_doc_idx
                    score_matrix_stage1[global_doc_idx][query_idx] = score

            # Aggregate scores across queries
            doc_scores_stage1 = []
            for doc_idx in range(num_docs):
                scores = score_matrix_stage1[doc_idx]
                if score_aggregation == "mean":
                    agg_score = sum(scores) / len(scores)
                else:  # max
                    agg_score = max(scores)

                doc_scores_stage1.append((doc_idx, agg_score))

            # Sort by score
            doc_scores_stage1.sort(key=lambda x: x[1], reverse=True)

            # Filter by min_score FIRST, then take top-K
            filtered_scores = [(idx, score) for idx, score in doc_scores_stage1 if score >= self.stage1_min_score]
            num_filtered_by_score = len(doc_scores_stage1) - len(filtered_scores)

            # Take top-K from filtered set
            top_k_indices = [doc_idx for doc_idx, _ in filtered_scores[: self.stage1_top_k]]
            top_k_documents = [documents[idx] for idx in top_k_indices]

            # Log score distribution for Stage 1
            from ..query.kg.helpers import log_score_distribution

            # Map indices back to actual chunks for logging
            stage1_chunks = []
            for idx, score in doc_scores_stage1:
                chunk = documents[idx]
                stage1_chunks.append(
                    {
                        "id": chunk.get("id") or chunk.get("chunk_id", f"idx_{idx}"),
                        "chunk_id": chunk.get("id") or chunk.get("chunk_id", f"idx_{idx}"),
                        "content": chunk.get("content") or chunk.get("text", ""),
                        "score": score,
                        "rerank_score": score,
                    }
                )
            log_score_distribution(stage1_chunks, "rerank_score", "Stage 1 Reranking (TinyBERT)")

            stage1_elapsed = time.time() - stage1_start
            if num_filtered_by_score > 0:
                logger.info(
                    f"[Stage 1] Complete in {stage1_elapsed:.3f}s: "
                    f"filtered {num_docs} → {len(filtered_scores)} (by min_score={self.stage1_min_score:.2f}) "
                    f"→ {len(top_k_documents)} (top-K={self.stage1_top_k}) "
                    f"(top score: {doc_scores_stage1[0][1]:.4f})"
                )
            else:
                logger.info(
                    f"[Stage 1] Complete in {stage1_elapsed:.3f}s: "
                    f"filtered {num_docs} → {len(top_k_documents)} documents "
                    f"(top score: {doc_scores_stage1[0][1]:.4f})"
                )

            # Detailed logging for Stage 1
            if detailed_logger:
                # Log each chunk with score and rank to rerank_stage1.jsonl
                for rank, (doc_idx, score) in enumerate(doc_scores_stage1, 1):
                    chunk_id = documents[doc_idx].get("id") or documents[doc_idx].get("chunk_id", f"doc-{doc_idx}")
                    detailed_logger.log_rerank_stage1_chunk(
                        {
                            "chunk_id": chunk_id,
                            "score": float(score),
                            "rank": rank,
                        }
                    )

                # Compute score distribution statistics for summary
                scores_array = [score for _, score in doc_scores_stage1]
                if scores_array:
                    import numpy as np

                    stage1_summary = {
                        "model": self.stage1_config.get("model", "cross-encoder/ms-marco-TinyBERT-L2-v2"),
                        "input_chunks": num_docs,
                        "output_chunks": len(top_k_documents),
                        "timing_ms": int(stage1_elapsed * 1000),
                        "score_distribution": {
                            "mean": float(np.mean(scores_array)),
                            "std": float(np.std(scores_array)),
                            "min": float(np.min(scores_array)),
                            "max": float(np.max(scores_array)),
                            "median": float(np.median(scores_array)),
                            "percentiles": {
                                "p25": float(np.percentile(scores_array, 25)),
                                "p50": float(np.percentile(scores_array, 50)),
                                "p75": float(np.percentile(scores_array, 75)),
                                "p90": float(np.percentile(scores_array, 90)),
                                "p95": float(np.percentile(scores_array, 95)),
                            },
                        },
                    }
                    detailed_logger.log_rerank_stage1_summary(stage1_summary)

        # ===== STAGE 2: Precise ranking of top-K =====
        stage2_start = time.time()
        num_stage2_docs = len(top_k_documents)
        logger.info(f"[Stage 2] Ranking top {num_stage2_docs} candidates...")

        # Initialize score matrix for stage 2
        score_matrix_stage2 = [[0.0] * num_queries for _ in range(num_stage2_docs)]

        # Split documents across workers for parallel processing
        docs_per_worker_s2 = (num_stage2_docs + self.stage2_num_workers - 1) // self.stage2_num_workers

        # Submit tasks: split documents across workers for each query
        tasks_submitted = 0
        task_map = {}  # task_id -> (query_idx, worker_id)

        for query_idx, query in enumerate(queries):
            for worker_id in range(self.stage2_num_workers):
                # Calculate document slice for this worker
                start_idx = worker_id * docs_per_worker_s2
                end_idx = min(start_idx + docs_per_worker_s2, num_stage2_docs)

                if start_idx >= num_stage2_docs:
                    break  # No more documents for this worker

                doc_slice = top_k_documents[start_idx:end_idx]
                task_id = f"{uuid.uuid4().hex[:8]}-s2-q{query_idx}-w{worker_id}"
                task_map[task_id] = (query_idx, worker_id)

                self.stage2_task_queue.put((task_id, query, doc_slice))
                tasks_submitted += 1

        logger.info(
            f"[Stage 2] Submitted {tasks_submitted} tasks to {self.stage2_num_workers} processes "
            f"({num_queries} queries × ~{self.stage2_num_workers} workers, {docs_per_worker_s2} docs/worker)"
        )

        # Collect stage 2 results
        for _ in range(tasks_submitted):
            task_id, indexed_scores = self.stage2_result_queue.get()

            # Get query index and worker ID
            query_idx, worker_id = task_map[task_id]

            # Calculate document offset for this worker's slice
            doc_offset = worker_id * docs_per_worker_s2

            # Store scores in matrix (adjust indices for document slice)
            for local_doc_idx, score in indexed_scores:
                global_doc_idx = doc_offset + local_doc_idx
                score_matrix_stage2[global_doc_idx][query_idx] = score

        # Aggregate scores across queries
        doc_scores_stage2 = []
        for doc_idx in range(num_stage2_docs):
            scores = score_matrix_stage2[doc_idx]
            if score_aggregation == "mean":
                agg_score = sum(scores) / len(scores)
            else:  # max
                agg_score = max(scores)

            doc_scores_stage2.append((doc_idx, agg_score))

        # Sort by score
        doc_scores_stage2.sort(key=lambda x: x[1], reverse=True)

        # Build final ranked results with min_score filtering
        ranked_docs = []
        for doc_idx, score in doc_scores_stage2:
            # Filter by stage2_min_score
            if score >= self.stage2_min_score:
                doc = top_k_documents[doc_idx].copy()
                doc["rerank_score"] = score
                ranked_docs.append(doc)

        # Log score distribution for Stage 2
        from ..query.kg.helpers import log_score_distribution

        log_score_distribution(ranked_docs, "rerank_score", "Stage 2 Reranking (Jina - Final)")

        stage2_elapsed = time.time() - stage2_start
        overall_elapsed = time.time() - overall_start

        num_filtered_stage2 = num_stage2_docs - len(ranked_docs)
        top_score_str = f"{doc_scores_stage2[0][1]:.4f}" if doc_scores_stage2 else "N/A"
        if num_stage2_docs == 0:
            logger.info(
                f"[Stage 2] Complete in {stage2_elapsed:.3f}s: "
                f"no documents to rank (all filtered by Stage 1)"
            )
        elif num_filtered_stage2 > 0:
            logger.info(
                f"[Stage 2] Complete in {stage2_elapsed:.3f}s: "
                f"ranked {num_stage2_docs} → {len(ranked_docs)} documents "
                f"(filtered {num_filtered_stage2} by min_score={self.stage2_min_score:.2f}) "
                f"(top score: {top_score_str})"
            )
        else:
            logger.info(
                f"[Stage 2] Complete in {stage2_elapsed:.3f}s: "
                f"ranked {num_stage2_docs} documents "
                f"(top score: {top_score_str})"
            )

        # Detailed logging for Stage 2
        if detailed_logger:
            # Log each chunk with score, rank, and kept status to rerank_stage2.jsonl
            num_kept = 0
            for rank, (doc_idx, score) in enumerate(doc_scores_stage2, 1):
                doc = top_k_documents[doc_idx]
                chunk_id = doc.get("id") or doc.get("chunk_id", f"doc-{doc_idx}")
                kept = score >= self.stage2_min_score
                if kept:
                    num_kept += 1

                detailed_logger.log_rerank_stage2_chunk(
                    {
                        "chunk_id": chunk_id,
                        "score": float(score),
                        "rank": rank,
                        "kept": kept,
                    }
                )

            # Compute score distribution statistics for summary
            scores_array = [score for _, score in doc_scores_stage2]
            if scores_array:
                import numpy as np

                stage2_summary = {
                    "model": self.stage2_config.get("model", "jinaai/jina-reranker-v2-base-multilingual"),
                    "input_chunks": num_stage2_docs,
                    "output_chunks": len(ranked_docs),
                    "filtered_by_min_score": num_kept,
                    "min_score_threshold": self.stage2_min_score,
                    "timing_ms": int(stage2_elapsed * 1000),
                    "score_distribution": {
                        "mean": float(np.mean(scores_array)),
                        "std": float(np.std(scores_array)),
                        "min": float(np.min(scores_array)),
                        "max": float(np.max(scores_array)),
                        "median": float(np.median(scores_array)),
                        "percentiles": {
                            "p25": float(np.percentile(scores_array, 25)),
                            "p50": float(np.percentile(scores_array, 50)),
                            "p75": float(np.percentile(scores_array, 75)),
                            "p90": float(np.percentile(scores_array, 90)),
                            "p95": float(np.percentile(scores_array, 95)),
                        },
                    },
                }
                detailed_logger.log_rerank_stage2_summary(stage2_summary)
        # Format final log message
        stage1_filtered = self.stage1_top_k if num_docs > self.stage1_top_k else num_docs
        speedup_text = (
            f"\n  Speedup estimate: ~{num_docs / self.stage1_top_k:.1f}x vs single-stage"
            if num_docs > self.stage1_top_k
            else ""
        )

        logger.info(
            f"=== TWO-STAGE RERANKING COMPLETE ==="
            f"\n  Total time: {overall_elapsed:.3f}s"
            f"\n  Stage 1 (filtering): {stage1_elapsed:.3f}s "
            f"({stage1_elapsed / overall_elapsed * 100:.1f}%)"
            f"\n  Stage 2 (ranking): {stage2_elapsed:.3f}s "
            f"({stage2_elapsed / overall_elapsed * 100:.1f}%)"
            f"\n  Documents processed: {num_docs} → {stage1_filtered} → {len(ranked_docs)}"
            f"{speedup_text}"
        )

        return ranked_docs
