"""
Neo4j graph storage adapter for RAG query system.

Handles:
- Entity node retrieval
- Relationship traversal
- Graph degree calculation

Adapted from: lightrag/kg/neo4j_impl.py
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .base import BaseGraphStorage

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
except ImportError:
    raise ImportError("neo4j not installed. Run: pip install neo4j>=5.0.0") from None

logger = logging.getLogger(__name__)


@dataclass
class Neo4jStorage(BaseGraphStorage):
    """
    Neo4j graph storage implementation for query operations.

    Node Schema:
    - Labels: {workspace} (primary) + entity type (Gene, Disease, etc.)
    - Properties: entity_id, entity_type, description

    Relationship Schema:
    - Type: DIRECTED (undirected, stored once)
    - Properties: weight, description
    """

    _driver: AsyncDriver | None = field(default=None, init=False, repr=False)
    _database: str | None = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    async def initialize(self) -> None:
        """Initialize Neo4j connection."""
        if self._initialized:
            return

        uri = self.config.get("uri", "bolt://localhost:7687")
        username = self.config.get("username", "neo4j")
        password = self.config.get("password")
        database = self.config.get("database")  # Optional, use default if None
        max_pool_size = self.config.get("max_connection_pool_size", 100)
        timeout = self.config.get("connection_timeout", 30.0)

        if not password:
            raise ValueError("Neo4j password is required in config")

        try:
            # Initialize Neo4j driver
            self._driver = AsyncGraphDatabase.driver(
                uri,
                auth=(username, password),
                max_connection_pool_size=max_pool_size,
                connection_timeout=timeout,
            )

            # Determine database name
            self._database = database if database else None

            # Test connection
            async with self._driver.session(database=self._database) as session:
                result = await session.run("RETURN 1 as test")
                await result.consume()

            logger.info(f"[{self.workspace}] Connected to Neo4j: {uri}, database: {self._database or 'default'}")
            self._initialized = True

        except Exception as e:
            logger.error(f"[{self.workspace}] Failed to initialize Neo4j: {e}")
            raise

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            try:
                await self._driver.close()
                logger.info(f"[{self.workspace}] Closed Neo4j connection")
            except Exception as e:
                logger.warning(f"[{self.workspace}] Error closing Neo4j: {e}")
            finally:
                self._driver = None
                self._initialized = False

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """
        Get a single node by entity_id.

        Args:
            node_id: The entity_id (e.g., "GENE:BRCA1")

        Returns:
            Dict with node data or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            query = f"""
            MATCH (n:`{self.workspace}` {{entity_id: $node_id}})
            RETURN n.entity_id as entity_id,
                   n.entity_type as entity_type,
                   n.description as description
            """

            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, node_id=node_id)
                record = await result.single()

                if record:
                    return dict(record)
                return None

        except Exception as e:
            logger.error(f"[{self.workspace}] get_node failed for {node_id}: {e}")
            raise

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        """
        Get multiple nodes by entity_ids (batch operation).

        Args:
            node_ids: List of entity_ids

        Returns:
            Dict mapping entity_id -> node data
        """
        if not self._initialized:
            await self.initialize()

        if not node_ids:
            return {}

        try:
            query = f"""
            MATCH (n:`{self.workspace}`)
            WHERE n.entity_id IN $node_ids
            RETURN n.entity_id as entity_id,
                   n.entity_type as entity_type,
                   n.description as description
            """

            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, node_ids=node_ids)
                records = await result.data()

                # Build dict mapping
                nodes = {}
                for record in records:
                    entity_id = record["entity_id"]
                    nodes[entity_id] = record

                logger.debug(f"[{self.workspace}] get_nodes_batch returned {len(nodes)}/{len(node_ids)} nodes")

                return nodes

        except Exception as e:
            logger.error(f"[{self.workspace}] get_nodes_batch failed: {e}")
            raise

    async def get_neighbors(self, node_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get neighbors of a node (connected via DIRECTED relationships).

        Args:
            node_id: The entity_id of the source node
            limit: Maximum number of neighbors to return

        Returns:
            List of dicts with relationship data
        """
        if not self._initialized:
            await self.initialize()

        try:
            query = f"""
            MATCH (source:`{self.workspace}` {{entity_id: $node_id}})-[r:DIRECTED]-(target)
            RETURN source.entity_id as src_id,
                   target.entity_id as tgt_id,
                   r.weight as weight,
                   r.description as description
            LIMIT $limit
            """

            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, node_id=node_id, limit=limit)
                records = await result.data()

                logger.debug(f"[{self.workspace}] get_neighbors({node_id}) returned {len(records)} relationships")

                return records

        except Exception as e:
            logger.error(f"[{self.workspace}] get_neighbors failed for {node_id}: {e}")
            raise

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Get degree (number of connections) for multiple nodes.

        Args:
            node_ids: List of entity_ids

        Returns:
            Dict mapping entity_id -> degree count
        """
        if not self._initialized:
            await self.initialize()

        if not node_ids:
            return {}

        try:
            query = f"""
            MATCH (n:`{self.workspace}`)-[r:DIRECTED]-()
            WHERE n.entity_id IN $node_ids
            RETURN n.entity_id as entity_id, count(r) as degree
            """

            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, node_ids=node_ids)
                records = await result.data()

                # Build dict mapping
                degrees = {}
                for record in records:
                    entity_id = record["entity_id"]
                    degrees[entity_id] = record["degree"]

                # Fill in missing nodes with degree 0
                for node_id in node_ids:
                    if node_id not in degrees:
                        degrees[node_id] = 0

                logger.debug(f"[{self.workspace}] node_degrees_batch returned {len(degrees)} degrees")

                return degrees

        except Exception as e:
            logger.error(f"[{self.workspace}] node_degrees_batch failed: {e}")
            raise

    async def node_degree(self, node_id: str) -> int:
        """
        Get degree (number of connections) for a single node.

        Args:
            node_id: The entity_id

        Returns:
            Degree count (number of connected edges)
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use batch method for consistency
            degrees = await self.node_degrees_batch([node_id])
            return degrees.get(node_id, 0)

        except Exception as e:
            logger.error(f"[{self.workspace}] node_degree failed for {node_id}: {e}")
            raise

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """
        Get all edges connected to a node.

        Args:
            source_node_id: The entity_id of the node

        Returns:
            List of (source_id, target_id) tuples representing edges,
            or None if the node doesn't exist
        """
        if not self._initialized:
            await self.initialize()

        try:
            query = f"""
            MATCH (n:`{self.workspace}` {{entity_id: $entity_id}})
            OPTIONAL MATCH (n)-[r:DIRECTED]-(connected:`{self.workspace}`)
            WHERE connected.entity_id IS NOT NULL
            WITH n, r, connected,
                 CASE WHEN startNode(r).entity_id = n.entity_id
                      THEN n.entity_id
                      ELSE connected.entity_id
                 END AS start_id,
                 CASE WHEN startNode(r).entity_id = n.entity_id
                      THEN connected.entity_id
                      ELSE n.entity_id
                 END AS end_id
            RETURN start_id, end_id
            """

            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, entity_id=source_node_id)
                records = await result.data()

                if not records:
                    # Check if node exists
                    node = await self.get_node(source_node_id)
                    if node is None:
                        return None
                    return []  # Node exists but has no edges

                edges = []
                for record in records:
                    start_id = record.get("start_id")
                    end_id = record.get("end_id")
                    if start_id and end_id:
                        edges.append((start_id, end_id))

                logger.debug(f"[{self.workspace}] get_node_edges({source_node_id}) returned {len(edges)} edges")

                return edges

        except Exception as e:
            logger.error(f"[{self.workspace}] get_node_edges failed for {source_node_id}: {e}")
            raise

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """
        Get edges for multiple nodes (batch operation).

        Args:
            node_ids: List of entity_ids

        Returns:
            Dict mapping entity_id -> list of (src_id, tgt_id) tuples
        """
        if not self._initialized:
            await self.initialize()

        if not node_ids:
            return {}

        try:
            query = f"""
            UNWIND $node_ids AS id
            MATCH (n:`{self.workspace}` {{entity_id: id}})
            OPTIONAL MATCH (n)-[r:DIRECTED]-(connected:`{self.workspace}`)
            RETURN id AS queried_id,
                   n.entity_id AS node_entity_id,
                   connected.entity_id AS connected_entity_id,
                   startNode(r).entity_id AS start_entity_id
            """

            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, node_ids=node_ids)
                records = await result.data()

                # Initialize dict with empty lists
                edges_dict = {node_id: [] for node_id in node_ids}

                # Process results
                for record in records:
                    queried_id = record["queried_id"]
                    node_entity_id = record["node_entity_id"]
                    connected_entity_id = record["connected_entity_id"]
                    start_entity_id = record["start_entity_id"]

                    # Skip if no connection
                    if not node_entity_id or not connected_entity_id:
                        continue

                    # Determine edge direction
                    if start_entity_id == node_entity_id:
                        # Outgoing edge
                        edges_dict[queried_id].append((node_entity_id, connected_entity_id))
                    else:
                        # Incoming edge
                        edges_dict[queried_id].append((connected_entity_id, node_entity_id))

                logger.debug(f"[{self.workspace}] get_nodes_edges_batch returned edges for {len(edges_dict)} nodes")

                return edges_dict

        except Exception as e:
            logger.error(f"[{self.workspace}] get_nodes_edges_batch failed: {e}")
            raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """
        Get the combined degree of an edge (sum of degrees of source and target nodes).

        Args:
            src_id: Source entity_id
            tgt_id: Target entity_id

        Returns:
            Sum of degrees of both nodes
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use batch method for consistency
            degrees = await self.edge_degrees_batch([(src_id, tgt_id)])
            return degrees.get((src_id, tgt_id), 0)

        except Exception as e:
            logger.error(f"[{self.workspace}] edge_degree failed for {src_id}-{tgt_id}: {e}")
            raise

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """
        Get combined degrees for multiple edges (batch operation).

        Args:
            edge_pairs: List of (src_id, tgt_id) tuples

        Returns:
            Dict mapping (src_id, tgt_id) -> combined degree
        """
        if not self._initialized:
            await self.initialize()

        if not edge_pairs:
            return {}

        try:
            # Collect unique node IDs from all edge pairs
            unique_node_ids = set()
            for src, tgt in edge_pairs:
                unique_node_ids.add(src)
                unique_node_ids.add(tgt)

            # Get degrees for all nodes in one query
            node_degrees = await self.node_degrees_batch(list(unique_node_ids))

            # Sum up degrees for each edge pair
            edge_degrees = {}
            for src, tgt in edge_pairs:
                edge_degrees[(src, tgt)] = node_degrees.get(src, 0) + node_degrees.get(tgt, 0)

            logger.debug(f"[{self.workspace}] edge_degrees_batch returned degrees for {len(edge_degrees)} edges")

            return edge_degrees

        except Exception as e:
            logger.error(f"[{self.workspace}] edge_degrees_batch failed: {e}")
            raise

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """
        Get edge properties between two nodes.

        Args:
            source_node_id: Source entity_id
            target_node_id: Target entity_id

        Returns:
            Dict with edge properties or None if edge doesn't exist
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Query for undirected edge (matches both directions)
            query = f"""
            MATCH (source:`{self.workspace}` {{entity_id: $src_id}})
                  -[r:DIRECTED]-
                  (target:`{self.workspace}` {{entity_id: $tgt_id}})
            RETURN r.weight as weight,
                   r.description as description,
                   r.keywords as keywords
            LIMIT 1
            """

            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, src_id=source_node_id, tgt_id=target_node_id)
                record = await result.single()

                if record:
                    return {
                        "weight": record.get("weight"),
                        "description": record.get("description"),
                        "keywords": record.get("keywords"),
                    }
                return None

        except Exception as e:
            logger.error(f"[{self.workspace}] get_edge failed for {source_node_id}-{target_node_id}: {e}")
            raise

    async def get_edges_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
        """
        Get edge properties for multiple edges in a single batch query.

        This is a batched version of get_edge() that avoids N+1 query problem.
        Instead of N separate queries, executes 1 query using UNWIND.

        Args:
            edge_pairs: List of (source_node_id, target_node_id) tuples

        Returns:
            Dict mapping (src_id, tgt_id) to edge properties dict
            Missing edges are not included in the result
        """
        if not edge_pairs:
            return {}

        if not self._initialized:
            await self.initialize()

        try:
            # Build list of [src, tgt] pairs for UNWIND
            edge_list = [[src, tgt] for src, tgt in edge_pairs]

            # Single query to fetch all edges at once using UNWIND
            query = f"""
            UNWIND $edge_pairs AS edge_pair
            MATCH (source:`{self.workspace}` {{entity_id: edge_pair[0]}})
                  -[r:DIRECTED]-
                  (target:`{self.workspace}` {{entity_id: edge_pair[1]}})
            RETURN edge_pair[0] as src_id,
                   edge_pair[1] as tgt_id,
                   r.weight as weight,
                   r.description as description,
                   r.keywords as keywords
            """

            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, edge_pairs=edge_list)

                edges = {}
                async for record in result:
                    edge_key = (record["src_id"], record["tgt_id"])
                    edges[edge_key] = {
                        "weight": record.get("weight"),
                        "description": record.get("description"),
                        "keywords": record.get("keywords"),
                    }

                logger.debug(f"[{self.workspace}] get_edges_batch returned {len(edges)}/{len(edge_pairs)} edges")
                return edges

        except Exception as e:
            logger.error(f"[{self.workspace}] get_edges_batch failed for {len(edge_pairs)} edges: {e}")
            raise
