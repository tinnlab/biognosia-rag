"""
Base storage interfaces for RAG query system.

These interfaces define the contract for vector, key-value, and graph storage.
Query-only operations - no insert/update methods.

Adapted from: lightrag/base.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseVectorStorage(ABC):
    """
    Abstract base class for vector storage operations.

    Used for:
    - Entities vector search (Milvus: {workspace}_entities)
    - Relationships vector search (Milvus: {workspace}_relationships)
    - Chunks vector search (Milvus: {workspace}_chunks)
    """

    workspace: str
    collection_name: str
    config: dict[str, Any]

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection."""
        pass

    @abstractmethod
    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        query_embedding: list[float] | None = None,
        partition_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query the vector storage for similar items.

        Args:
            query_text: The query string (will be embedded if query_embedding not provided)
            top_k: Number of top results to return
            query_embedding: Optional pre-computed embedding (for performance)
            partition_names: Optional list of partitions to search (Milvus entities only)

        Returns:
            List of dicts containing:
            - id: Document ID
            - entity_name: Entity name (for entities collection)
            - src_id, tgt_id: Relationship endpoints (for relationships collection)
            - full_doc_id: Document ID (for chunks collection)
            - content: Text content (if available)
            - score: Similarity score
            - Other metadata fields
        """
        pass

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """
        Get a single item by its ID.

        Args:
            id: The unique identifier

        Returns:
            Dict with item data, or None if not found
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Get multiple items by their IDs.

        Args:
            ids: List of unique identifiers

        Returns:
            List of dicts with item data (may be shorter than input if some IDs not found)
        """
        pass

    @abstractmethod
    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """
        Get embeddings for multiple items by their IDs.

        Args:
            ids: List of unique identifiers

        Returns:
            Dict mapping ID -> embedding vector
            Example: {"chunk-1": [0.1, 0.2, ...], "chunk-2": [0.3, 0.4, ...]}
        """
        pass


@dataclass
class BaseKVStorage(ABC):
    """
    Abstract base class for key-value storage operations.

    Used for:
    - Text chunks content (Redis: {workspace}_text_chunks)
    - Entity sources (Redis: {workspace}_entity_sources)
    - Chunk entities (Redis: {workspace}_chunk_entities)
    - Document status (Redis: {workspace}_doc_status)
    """

    workspace: str
    namespace: str
    config: dict[str, Any]

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection."""
        pass

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """
        Get value by ID.

        Args:
            id: The unique identifier

        Returns:
            Dict with data, or None if not found
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Get values by multiple IDs.

        Args:
            ids: List of unique identifiers

        Returns:
            List of dicts with data (may be shorter than input if some IDs not found)
        """
        pass

    @abstractmethod
    async def get_set_members(self, key: str) -> set[str]:
        """
        Get members of a Redis set.

        Args:
            key: The set key

        Returns:
            Set of member strings
        """
        pass


@dataclass
class BaseGraphStorage(ABC):
    """
    Abstract base class for graph storage operations.

    Used for:
    - Entity node retrieval (Neo4j: {workspace} nodes)
    - Relationship traversal (Neo4j: DIRECTED edges)
    """

    workspace: str
    config: dict[str, Any]

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection."""
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """
        Get a single node by its entity_id.

        Args:
            node_id: The entity_id (e.g., "GENE:BRCA1")

        Returns:
            Dict with node data:
            - entity_id: Entity identifier
            - entity_type: Entity type
            - description: Entity description
            Or None if not found
        """
        pass

    @abstractmethod
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        """
        Get multiple nodes by their entity_ids (batch operation).

        Args:
            node_ids: List of entity_ids

        Returns:
            Dict mapping entity_id -> node data
            Example: {"GENE:BRCA1": {"entity_id": "GENE:BRCA1", ...}, ...}
        """
        pass

    @abstractmethod
    async def get_neighbors(self, node_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get neighbors of a node (connected via DIRECTED relationships).

        Args:
            node_id: The entity_id of the source node
            limit: Maximum number of neighbors to return

        Returns:
            List of dicts with relationship data:
            - src_id: Source entity_id
            - tgt_id: Target entity_id
            - weight: Relationship weight (co-occurrence count)
            - description: Relationship description
        """
        pass

    @abstractmethod
    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Get degree (number of connections) for multiple nodes.

        Args:
            node_ids: List of entity_ids

        Returns:
            Dict mapping entity_id -> degree count
            Example: {"GENE:BRCA1": 15, "DISEASE:Cancer": 42, ...}
        """
        pass

    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        """
        Get degree (number of connections) for a single node.

        Args:
            node_id: The entity_id

        Returns:
            Degree count (number of connected edges)
        """
        pass

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """
        Get all edges connected to a node.

        Args:
            source_node_id: The entity_id of the node

        Returns:
            List of (source_id, target_id) tuples representing edges,
            or None if the node doesn't exist
        """
        pass

    @abstractmethod
    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """
        Get edges for multiple nodes (batch operation).

        Args:
            node_ids: List of entity_ids

        Returns:
            Dict mapping entity_id -> list of (src_id, tgt_id) tuples
            Example: {"GENE:BRCA1": [("GENE:BRCA1", "GO:0006281"), ...], ...}
        """
        pass

    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """
        Get the combined degree of an edge (sum of degrees of source and target nodes).

        Args:
            src_id: Source entity_id
            tgt_id: Target entity_id

        Returns:
            Sum of degrees of both nodes
        """
        pass

    @abstractmethod
    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """
        Get combined degrees for multiple edges (batch operation).

        Args:
            edge_pairs: List of (src_id, tgt_id) tuples

        Returns:
            Dict mapping (src_id, tgt_id) -> combined degree
            Example: {("GENE:BRCA1", "GO:0006281"): 25, ...}
        """
        pass

    @abstractmethod
    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """
        Get edge properties between two nodes.

        Args:
            source_node_id: Source entity_id
            target_node_id: Target entity_id

        Returns:
            Dict with edge properties:
            - weight: Relationship weight
            - description: Relationship description
            - keywords: Related keywords
            Or None if edge doesn't exist
        """
        pass
