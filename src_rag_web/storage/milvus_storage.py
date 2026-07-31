"""
Milvus vector storage adapter for RAG query system.

Handles:
- Vector similarity search (entities, relationships, chunks)
- Batch retrieval by IDs
- Vector embedding retrieval

Adapted from: lightrag/kg/milvus_impl.py
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from .base import BaseVectorStorage

try:
    from pymilvus import MilvusClient
except ImportError:
    raise ImportError("pymilvus not installed. Run: pip install pymilvus==2.5.2") from None

logger = logging.getLogger(__name__)


@dataclass
class MilvusStorage(BaseVectorStorage):
    """
    Milvus vector storage implementation for query operations.

    Collections:
    - {workspace}_entities: Entity vectors with partitions by type
    - {workspace}_relationships: Relationship vectors
    - {workspace}_chunks: Text chunk vectors
    """

    _client: MilvusClient | None = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _entity_collections: list[str] = field(default_factory=list, init=False, repr=False)

    async def initialize(self) -> None:
        """Initialize Milvus connection and discover entity collections."""
        if self._initialized:
            return

        host = self.config.get("host", "localhost")
        port = self.config.get("port", "19530")
        timeout = self.config.get("timeout", 30)

        try:
            # Initialize Milvus client
            uri = f"http://{host}:{port}"
            self._client = MilvusClient(uri=uri, timeout=timeout)

            # Build full collection name
            self.full_collection_name = f"{self.workspace}_{self.collection_name}"

            # Verify collection exists
            if not self._client.has_collection(self.full_collection_name):
                raise ValueError(f"Collection '{self.full_collection_name}' does not exist in Milvus")

            # Load collection into memory
            self._client.load_collection(self.full_collection_name)

            # Discover entity collections (entities_*) once during initialization
            all_collections = self._client.list_collections()
            self._entity_collections = [c for c in all_collections if c.startswith("entities_")]
            logger.info(
                f"[{self.workspace}] Discovered {len(self._entity_collections)} entity collections: "
                f"{self._entity_collections}"
            )

            logger.info(f"[{self.workspace}] Connected to Milvus: {uri}, collection: {self.full_collection_name}")
            self._initialized = True

        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to initialize Milvus: {e}")
            raise

    async def close(self) -> None:
        """Close Milvus connection."""
        if self._client:
            try:
                self._client.close()
                logger.info(f"[{self.workspace}] Closed Milvus connection")
            except Exception as e:
                logger.warning(f"[{self.workspace}] Error closing Milvus: {e}")
            finally:
                self._client = None
                self._initialized = False
                self._entity_collections = []

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        query_embedding: list[float] | None = None,
        partition_names: list[str] | None = None,
        cosine_threshold: float = 0.0,
        search_params: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query Milvus for similar vectors.

        Args:
            query_text: Query string (for logging only if embedding provided)
            top_k: Number of top results
            query_embedding: Pre-computed query embedding (REQUIRED for now)
            partition_names: Optional partitions to search (for entities collection)
            cosine_threshold: Minimum cosine similarity threshold (default: 0.0)
            search_params: Optional search parameters (default: COSINE with ef=max(top_k*2, 200))

        Returns:
            List of results with all fields + similarity score
        """
        if not self._initialized:
            await self.initialize()

        if query_embedding is None:
            raise ValueError("query_embedding is required. Compute embedding before calling query().")

        try:
            # Search parameters
            # Milvus HNSW constraint: ef must be >= k (top_k)
            if search_params is None:
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"ef": max(top_k * 2, 200)},  # HNSW search parameter
                }

            # Perform search (wrapped in to_thread for true async parallelism)
            # Milvus SDK search() is blocking, so we run it in thread pool
            results = await asyncio.to_thread(
                self._client.search,
                collection_name=self.full_collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["*"],
                search_params=search_params,
                partition_names=partition_names,
            )

            # Format results
            formatted_results = []
            for hit in results[0]:  # results[0] because we query with one vector
                score = hit.get("distance", 0.0)  # Cosine similarity

                # Apply cosine threshold filter
                if score < cosine_threshold:
                    continue

                result = {
                    "id": hit.get("id"),
                    "score": score,
                }

                # Extract fields from the 'entity' dict (Milvus search results nest fields here)
                entity_data = hit.get("entity", {})
                for key, value in entity_data.items():
                    if key not in ["id", "vector"]:  # Skip id (already added) and vector (large)
                        result[key] = value

                formatted_results.append(result)

            logger.debug(
                f"[{self.workspace}] Query returned {len(formatted_results)} results "
                f"(top_k={top_k}, threshold={cosine_threshold}, partitions={partition_names})"
            )

            return formatted_results

        except Exception as e:
            logger.error(f"[{self.workspace}] Milvus query failed: {e}")
            raise

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a single item by ID."""
        if not self._initialized:
            await self.initialize()

        try:
            results = self._client.query(
                collection_name=self.full_collection_name,
                filter=f'id == "{id}"',
                output_fields=["*"],
                limit=1,
            )

            if results:
                return results[0]
            return None

        except Exception as e:
            logger.error(f"[{self.workspace}] get_by_id failed for {id}: {e}")
            raise

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Get multiple items by IDs.

        Batches queries to stay under Milvus's 16,384 result limit.
        """
        if not self._initialized:
            await self.initialize()

        if not ids:
            return []

        try:
            # Milvus has a hard limit of 16,384 results per query
            # Batch the queries to stay under this limit
            MAX_BATCH_SIZE = 16000  # Leave some margin
            all_results = []

            for i in range(0, len(ids), MAX_BATCH_SIZE):
                batch_ids = ids[i : i + MAX_BATCH_SIZE]

                # Build filter expression
                # Format: id in ["id1", "id2", "id3"]
                id_list_str = ", ".join([f'"{id}"' for id in batch_ids])
                filter_expr = f"id in [{id_list_str}]"

                results = self._client.query(
                    collection_name=self.full_collection_name,
                    filter=filter_expr,
                    output_fields=["*"],
                    limit=len(batch_ids),
                )

                all_results.extend(results)

            logger.debug(f"[{self.workspace}] get_by_ids returned {len(all_results)}/{len(ids)} items")

            return all_results

        except Exception as e:
            logger.error(f"[{self.workspace}] get_by_ids failed: {e}")
            raise

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """
        Get vectors for multiple IDs.

        Batches queries to stay under Milvus's 16,384 result limit.

        Returns:
            Dict mapping ID -> vector
        """
        if not self._initialized:
            await self.initialize()

        if not ids:
            return {}

        try:
            import time

            # Milvus has a hard limit of 16,384 results per query
            # Batch the queries to stay under this limit
            MAX_BATCH_SIZE = 16000  # Leave some margin
            vectors = {}

            num_batches = (len(ids) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            logger.info(
                f"[{self.workspace}] Fetching {len(ids)} vectors from Milvus in {num_batches} batches "
                f"(batch_size={MAX_BATCH_SIZE})"
            )

            total_start = time.time()
            for batch_idx, i in enumerate(range(0, len(ids), MAX_BATCH_SIZE), 1):
                batch_start = time.time()
                batch_ids = ids[i : i + MAX_BATCH_SIZE]

                # Build filter expression
                id_list_str = ", ".join([f'"{id}"' for id in batch_ids])
                filter_expr = f"id in [{id_list_str}]"

                results = self._client.query(
                    collection_name=self.full_collection_name,
                    filter=filter_expr,
                    output_fields=["id", "vector"],
                    limit=len(batch_ids),
                )

                # Build ID -> vector mapping
                for item in results:
                    vectors[item["id"]] = item["vector"]

                batch_elapsed = time.time() - batch_start
                logger.info(
                    f"[{self.workspace}] Batch {batch_idx}/{num_batches}: "
                    f"fetched {len(results)} vectors in {batch_elapsed:.3f}s"
                )

            total_elapsed = time.time() - total_start
            logger.info(
                f"[{self.workspace}] get_vectors_by_ids: {len(vectors)}/{len(ids)} vectors in {total_elapsed:.3f}s "
                f"(avg: {total_elapsed / num_batches:.3f}s/batch)"
            )

            return vectors

        except Exception as e:
            logger.error(f"[{self.workspace}] get_vectors_by_ids failed: {e}")
            raise

    async def search_entities(
        self,
        query_embeddings: list,
        top_k: int = 1,
        min_similarity: float = 0.85,
        exclude_collections: list[str] | None = None,
        include_collections: list[str] | None = None,
        search_field: str = "label_embedding",
    ) -> list[list[dict[str, Any]]]:
        """
        Search entity collections (entities_*) using specified embedding field.

        Uses entity collections discovered during initialization.
        Queries all collections in parallel using ThreadPoolExecutor.

        Args:
            query_embeddings: List of query embedding vectors
                            - 768-dim from MedCPT for label_embedding field
                            - 1024-dim from bge-m3 for vector field
            top_k: Number of top results per query per collection
            min_similarity: Minimum cosine similarity threshold (default: 0.85)
            exclude_collections: List of collection names to exclude (e.g., ["Taxonomy", "Disease"])
                                Collection names should match the suffix after "entities_"
            include_collections: List of collection names to include (e.g., ["Genes", "Disease"])
                                If specified, only these collections will be searched
                                Collection names should match the suffix after "entities_"
            search_field: Which embedding field to search ("label_embedding" or "vector")
                         - "label_embedding": 768-dim, for short entity names/labels
                         - "vector": 1024-dim, for longer content/descriptions

        Returns:
            List of result lists, one per query embedding.
            Each result contains entity_name, entity_type, and all dynamic fields.
        """
        if not self._initialized:
            await self.initialize()

        # Use cached entity collections list (discovered during initialization)
        if not self._entity_collections:
            logger.warning("No entity collections found (entities_*)")
            return [[] for _ in query_embeddings]

        # Filter collections based on include/exclude lists
        collections_to_search = self._entity_collections

        # Apply include filter first (if specified)
        if include_collections:
            # Convert include list to full collection names (entities_*)
            included_full_names = {f"entities_{name}" for name in include_collections}
            collections_to_search = [c for c in self._entity_collections if c in included_full_names]

            logger.debug(
                f"Filtered entity collections (include): {len(collections_to_search)}/{len(self._entity_collections)} "
                f"(included: {', '.join(include_collections)})"
            )

        # Apply exclude filter (if specified)
        if exclude_collections:
            # Convert exclude list to full collection names (entities_*)
            excluded_full_names = {f"entities_{name}" for name in exclude_collections}
            collections_to_search = [c for c in collections_to_search if c not in excluded_full_names]

            logger.debug(
                f"Filtered entity collections (exclude): {len(collections_to_search)} collections "
                f"(excluded: {', '.join(exclude_collections)})"
            )

        if not collections_to_search:
            logger.warning("No entity collections to search after filtering")
            return [[] for _ in query_embeddings]

        try:
            # Search parameters for COSINE similarity on label_embedding field
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 200},
            }

            # Convert embeddings to list format if needed
            query_data = []
            for emb in query_embeddings:
                if hasattr(emb, "tolist"):
                    query_data.append(emb.tolist())
                else:
                    query_data.append(list(emb))

            # Search all collections in parallel using ThreadPoolExecutor
            # This ensures true parallel execution across collections
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            loop = asyncio.get_event_loop()

            # Get max_workers from config (default: 10)
            max_workers = self.config.get("entity_search_max_workers", 10)
            # Use minimum of configured workers and number of collections
            num_collections = len(collections_to_search)
            actual_workers = min(num_collections, max_workers) if max_workers > 0 else num_collections

            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                # Submit all search tasks to thread pool
                futures = [
                    loop.run_in_executor(
                        executor,
                        self._search_single_collection_sync,
                        collection_name,
                        query_data,
                        top_k,
                        min_similarity,
                        search_params,
                        search_field,
                    )
                    for collection_name in collections_to_search
                ]

                # Wait for all searches to complete
                collection_results = await asyncio.gather(*futures, return_exceptions=True)

            # Initialize combined results for each query
            all_results = [[] for _ in query_embeddings]

            # Merge results from all collections
            for collection_result in collection_results:
                # Skip failed collections (exceptions)
                if isinstance(collection_result, Exception):
                    logger.warning(f"Collection search failed: {collection_result}")
                    continue

                # collection_result is a list of lists (one per query)
                for query_idx, query_results in enumerate(collection_result):
                    all_results[query_idx].extend(query_results)

            # Keep top_k results PER collection type (not globally)
            # This ensures we don't lose entities just because a different type had higher score
            for query_idx in range(len(all_results)):
                results_by_collection = {}
                for result in all_results[query_idx]:
                    collection = result.get("collection", "")
                    if collection not in results_by_collection:
                        results_by_collection[collection] = []
                    results_by_collection[collection].append(result)

                # For each collection, keep top_k matches above threshold
                combined_results = []
                for collection, coll_results in results_by_collection.items():
                    # Sort by similarity score (descending)
                    sorted_results = sorted(coll_results, key=lambda x: x["score"], reverse=True)
                    # Keep top_k per collection
                    combined_results.extend(sorted_results[:top_k])

                # Sort combined results by score
                all_results[query_idx] = sorted(combined_results, key=lambda x: x["score"], reverse=True)

            total_matches = sum(len(r) for r in all_results)
            logger.debug(
                f"Entity search across {len(collections_to_search)} collections: "
                f"{len(query_embeddings)} queries, {total_matches} total matches (min_similarity={min_similarity})"
            )

            return all_results

        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            # Return empty results for all queries
            return [[] for _ in query_embeddings]

    def _search_single_collection_sync(
        self,
        collection_name: str,
        query_data: list,
        top_k: int,
        min_similarity: float,
        search_params: dict,
        search_field: str = "label_embedding",
    ) -> list[list[dict[str, Any]]]:
        """
        Search a single entity collection synchronously (for ThreadPoolExecutor).

        This method is designed to run in a thread pool for true parallel execution.
        Each thread gets its own execution context, avoiding GIL issues with I/O operations.

        Args:
            collection_name: Name of the collection to search
            query_data: List of query embeddings (already converted to lists)
            top_k: Number of top results per query
            min_similarity: Minimum similarity threshold
            search_params: Milvus search parameters
            search_field: Which embedding field to search ("label_embedding" or "vector")

        Returns:
            List of result lists, one per query embedding.
        """
        try:
            # Load collection if needed
            try:
                self._client.load_collection(collection_name)
            except Exception:
                pass  # May already be loaded

            # Perform batch search on specified embedding field
            # This is a synchronous I/O operation that releases the GIL
            results = self._client.search(
                collection_name=collection_name,
                data=query_data,
                anns_field=search_field,  # Search on label_embedding or vector field
                limit=top_k,
                output_fields=["*"],  # Return all fields including entity_name and dynamic fields
                search_params=search_params,
            )

            # Initialize results for each query
            query_results = [[] for _ in query_data]

            # Format results for each query
            for query_idx, query_hits in enumerate(results):
                for hit in query_hits:
                    score = hit.get("distance", 0.0)  # Cosine similarity

                    # Apply similarity threshold
                    if score < min_similarity:
                        continue

                    # Extract all entity data from the hit
                    entity_data = hit.get("entity", {})
                    result = {
                        "id": hit.get("id"),  # ID is at top level, needed for chunk retrieval
                        "score": score,
                        "collection": collection_name,  # Track which collection this came from
                    }

                    # Add all fields except vector and label_embedding (too large)
                    for key, value in entity_data.items():
                        if key not in ["id", "vector", "label_embedding"]:  # Skip id (already added)
                            result[key] = value

                    query_results[query_idx].append(result)

            return query_results

        except Exception as e:
            logger.warning(f"Entity search failed for {collection_name}: {e}")
            # Return empty results for all queries
            return [[] for _ in query_data]
