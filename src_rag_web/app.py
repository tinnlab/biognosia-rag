"""
RAG Application entry point.

This module initializes all components and exposes a single query() function.
"""

import logging
from typing import Any

from .config import (
    check_endpoint_placeholders,
    load_config,
    require_llm_sdk,
    validate_llm_config,
    warn_if_hf_token_missing,
)
from .embedding import EmbeddingManager
from .entity_extraction import EntityStatisticsManager
from .llm import create_llm_provider
from .query import QueryParam, bypass_query, kg_query, naive_query
from .query.auto_mode import query_with_tools
from .query.entities_only import entities_only_query
from .query.worker_pool import RetrievalWorkerPool
from .rerank import RerankProcessor
from .rerank.worker_pool import RerankWorkerPool
from .storage import MilvusStorage, MongoStorage, Neo4jStorage, RedisStorage

logger = logging.getLogger(__name__)


class RAGApp:
    """
    RAG Application that manages all components.

    Usage:
        app = RAGApp(config_path="config/rag.conf")
        await app.initialize()
        result = await app.query("What is BRCA1?")
    """

    def __init__(self, config_path: str = "config/rag.conf", skip_llm: bool = False):
        """
        Initialize RAG application.

        Args:
            config_path: Path to configuration file
            skip_llm: If True, skip LLM provider initialization (for per-request LLM)
        """
        self.config_path = config_path
        self.skip_llm = skip_llm
        self.config: dict[str, Any] | None = None
        self.embedding_manager: EmbeddingManager | None = None
        self.llm_provider = None
        self.rerank_processor: RerankProcessor | None = None
        self.entity_stats_manager: EntityStatisticsManager | None = None

        # Storage instances
        self.chunks_vdb: MilvusStorage | None = None
        self.entities_vdb: MilvusStorage | None = None
        self.relationships_vdb: MilvusStorage | None = None
        self.text_chunks_storage: RedisStorage | None = None
        self.chunk_entity_relation_storage: RedisStorage | None = None
        self.graph_storage: Neo4jStorage | None = None
        self.es_client: Any | None = None
        self.mongo_storage: MongoStorage | None = None
        self.cache_manager: RedisStorage | None = None
        self.worker_pool: RetrievalWorkerPool | None = None
        self.rerank_worker_pool: RerankWorkerPool | None = None
        self.two_stage_rerank_pool: Any | None = None

        self._initialized = False

    async def initialize(self):
        """Initialize all components."""
        if self._initialized:
            logger.warning("App already initialized")
            return

        logger.info(f"Loading configuration from {self.config_path}")
        self.config = load_config(self.config_path)

        # Cheap checks before anything expensive: an unedited .env should fail
        # here, not after several minutes of loading models onto the GPU.
        check_endpoint_placeholders(self.config)
        warn_if_hf_token_missing()
        if not self.skip_llm:
            llm_config = self.config.get("llm", {})
            validate_llm_config(llm_config)
            require_llm_sdk(llm_config.get("provider", ""))

        # Initialize embedding manager
        logger.info("Initializing embedding manager...")
        embedding_config = self.config.get("embedding", {})
        self.embedding_manager = EmbeddingManager(embedding_config)

        # Preload embedding models (avoid lazy loading during queries)
        logger.info("Preloading embedding models...")
        self.embedding_manager.preload_models()

        # Initialize LLM provider (skip if per-request LLM is used)
        if self.skip_llm:
            logger.info("Skipping LLM provider initialization (per-request LLM mode)")
        else:
            logger.info("Initializing LLM provider...")
            llm_config = self.config.get("llm", {})
            self.llm_provider = create_llm_provider(llm_config)

        # Initialize reranking worker pool (BEFORE RerankProcessor)
        logger.info("Initializing reranking...")
        rerank_config = self.config.get("rerank", {})
        two_stage_config = self.config.get("rerank_two_stage", {})

        # Check if two-stage reranking is enabled
        two_stage_enabled = two_stage_config.get("enabled", False)

        if two_stage_enabled:
            # TWO-STAGE RERANKING: Create two pools (stage 1 + stage 2)
            from .rerank.two_stage_pool import TwoStageRerankerPool

            logger.info("Two-stage reranking enabled - creating cascade pools...")

            # Build stage 1 config (fast model)
            stage1_config = {
                "model": two_stage_config.get("stage1_model", "cross-encoder/ms-marco-TinyBERT-L2-v2"),
                "device": two_stage_config.get("stage1_device", "cuda:1"),
                "max_length": two_stage_config.get("stage1_max_length", 512),
                "batch_size": two_stage_config.get("stage1_batch_size", 4000),
                "num_workers": two_stage_config.get("stage1_num_workers", 10),
            }

            # Stage 2 uses main rerank config (precise model)
            stage2_config = rerank_config.copy()

            # Create and start two-stage pool
            self.two_stage_rerank_pool = TwoStageRerankerPool(
                stage1_config=stage1_config,
                stage2_config=stage2_config,
                two_stage_config=two_stage_config,
            )
            await self.two_stage_rerank_pool.start()

            # Create RerankProcessor with two-stage pool
            self.rerank_processor = RerankProcessor(
                config=rerank_config,
                worker_pool=None,
                two_stage_pool=self.two_stage_rerank_pool,
                two_stage_config=two_stage_config,
            )
            await self.rerank_processor.initialize()

        else:
            # SINGLE-STAGE RERANKING: Use regular worker pool
            num_workers = rerank_config.get("num_workers", 10)
            if num_workers > 0:
                logger.info(f"Starting rerank worker pool with {num_workers} workers...")
                self.rerank_worker_pool = RerankWorkerPool(rerank_config)
                await self.rerank_worker_pool.start()
            else:
                logger.info("Rerank worker pool disabled (num_workers=0)")
                self.rerank_worker_pool = None

            # Create RerankProcessor with single-stage worker pool
            self.rerank_processor = RerankProcessor(
                config=rerank_config,
                worker_pool=self.rerank_worker_pool,
                two_stage_pool=None,
                two_stage_config=None,
            )
            await self.rerank_processor.initialize()

        # Initialize entity statistics manager (for n-gram matching)
        ngram_config = self.config.get("ngram_entity_matching", {})
        if ngram_config.get("enable", False):
            stats_file = ngram_config.get("statistics_file")
            if stats_file:
                logger.info(f"Initializing entity statistics manager from {stats_file}")
                self.entity_stats_manager = EntityStatisticsManager(stats_file)
                self.entity_stats_manager.load_statistics()

        # Initialize Milvus storage
        logger.info("Initializing Milvus storage...")
        milvus_config = self.config.get("milvus", {})
        workspace = milvus_config.get("workspace", "lightrag")

        self.chunks_vdb = MilvusStorage(
            workspace=workspace,
            collection_name="chunks",
            config=milvus_config,
        )
        await self.chunks_vdb.initialize()

        self.entities_vdb = MilvusStorage(
            workspace=workspace,
            collection_name="entities",
            config=milvus_config,
        )
        await self.entities_vdb.initialize()

        # Relationships VDB may not exist
        try:
            self.relationships_vdb = MilvusStorage(
                workspace=workspace,
                collection_name="relationships",
                config=milvus_config,
            )
            await self.relationships_vdb.initialize()
        except Exception as e:
            logger.warning(f"Relationships collection not available: {e}")
            self.relationships_vdb = None

        # Initialize Redis storage
        logger.info("Initializing Redis storage...")
        redis_config = self.config.get("redis", {})

        self.text_chunks_storage = RedisStorage(
            workspace=workspace,
            namespace="text_chunks",  # Match insertion script namespace
            config=redis_config,
        )
        await self.text_chunks_storage.initialize()

        self.chunk_entity_relation_storage = RedisStorage(
            workspace=workspace,
            namespace="entity_sources",  # Redis keys: lightrag_entity_sources:ent-{hash}
            config=redis_config,
        )
        await self.chunk_entity_relation_storage.initialize()

        # Initialize Neo4j storage
        logger.info("Initializing Neo4j storage...")
        neo4j_config = self.config.get("neo4j", {})

        self.graph_storage = Neo4jStorage(
            workspace=workspace,
            config=neo4j_config,
        )
        await self.graph_storage.initialize()

        # Initialize Elasticsearch client (optional)
        es_config = self.config.get("elasticsearch", {})
        if es_config.get("enabled", False):
            logger.info("Initializing Elasticsearch client...")
            from elasticsearch import AsyncElasticsearch

            self.es_client = AsyncElasticsearch(
                hosts=[es_config.get("hosts", "http://localhost:9200")],
                basic_auth=(
                    (es_config.get("username"), es_config.get("password")) if es_config.get("username") else None
                ),
                request_timeout=es_config.get("timeout", 30),
                max_retries=es_config.get("max_retries", 3),
            )
            # Test connection
            try:
                await self.es_client.info()
                logger.info("Elasticsearch connection successful")
            except Exception as e:
                logger.error(f"Elasticsearch connection failed: {e}")
                await self.es_client.close()
                self.es_client = None

        # Initialize MongoDB storage (paper metadata for MCP annotations)
        mongo_config = self.config.get("mongodb", {})
        if mongo_config:
            logger.info("Initializing MongoDB storage...")
            self.mongo_storage = MongoStorage(mongo_config)
            try:
                await self.mongo_storage.initialize()
            except Exception as e:
                logger.warning(f"MongoDB initialization failed (annotations will use raw chunk text): {e}")
                self.mongo_storage = None

        # Initialize cache manager (Redis for keyword caching)
        self.cache_manager = RedisStorage(
            workspace=workspace,
            namespace="cache",
            config=redis_config,
        )
        await self.cache_manager.initialize()

        # Initialize and start persistent worker pool for parallel retrieval
        logger.info("Starting persistent worker pool (3 workers for KG/ES/Milvus)...")
        self.worker_pool = RetrievalWorkerPool()
        await self.worker_pool.start()

        self._initialized = True
        logger.info("RAG App initialized successfully")

    async def close(self):
        """Close all connections."""
        logger.info("Closing RAG App connections...")

        # Stop worker pools first
        if self.worker_pool:
            await self.worker_pool.stop()
        if self.rerank_worker_pool:
            await self.rerank_worker_pool.stop()
        if self.two_stage_rerank_pool:
            await self.two_stage_rerank_pool.stop()

        if self.chunks_vdb:
            await self.chunks_vdb.close()
        if self.entities_vdb:
            await self.entities_vdb.close()
        if self.relationships_vdb:
            await self.relationships_vdb.close()
        if self.text_chunks_storage:
            await self.text_chunks_storage.close()
        if self.chunk_entity_relation_storage:
            await self.chunk_entity_relation_storage.close()
        if self.graph_storage:
            await self.graph_storage.close()
        if self.es_client:
            await self.es_client.close()
        if self.mongo_storage:
            await self.mongo_storage.close()
        if self.cache_manager:
            await self.cache_manager.close()
        if self.llm_provider:
            await self.llm_provider.close()

        self._initialized = False
        logger.info("RAG App closed")

    async def query(
        self,
        query_text: str,
        mode: str | None = None,
        llm_provider=None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Query the RAG system.

        Args:
            query_text: User's query
            mode: Query mode (naive, local, global, hybrid, mix, bypass)
            llm_provider: Optional per-request LLM provider (overrides self.llm_provider)
            **kwargs: Additional query parameters (top_k, enable_rerank, etc.)

        Returns:
            Dictionary with response and metadata
        """
        if not self._initialized:
            raise RuntimeError("RAG App not initialized. Call await app.initialize() first.")

        # Use per-request provider if given, else fall back to app-level provider
        active_llm_provider = llm_provider or self.llm_provider
        if active_llm_provider is None:
            raise RuntimeError("No LLM provider available. Provide llm_provider or initialize with skip_llm=False.")

        # Get default mode from config if not specified
        if mode is None:
            auto_config = self.config.get("auto_mode", {})
            if auto_config.get("enabled", False):
                mode = "auto"
            else:
                mode = self.config.get("query", {}).get("default_mode", "mix")

        # Create query parameters
        query_config = self.config.get("query", {})
        param = QueryParam(
            mode=mode,
            top_k=kwargs.get("top_k", query_config.get("top_k", 10)),
            chunk_top_k=kwargs.get("chunk_top_k", query_config.get("chunk_top_k")),
            max_entity_tokens=kwargs.get("max_entity_tokens", query_config.get("max_entity_tokens", 2000)),
            max_relation_tokens=kwargs.get("max_relation_tokens", query_config.get("max_relation_tokens", 2000)),
            max_total_tokens=kwargs.get("max_total_tokens", query_config.get("max_total_tokens", 8000)),
            kg_chunk_pick_method=kwargs.get("kg_chunk_pick_method", query_config.get("kg_chunk_pick_method", "VECTOR")),
            kg_chunk_top_k=kwargs.get("kg_chunk_top_k", query_config.get("kg_chunk_top_k", 50)),
            max_related_chunks=kwargs.get("max_related_chunks", query_config.get("max_related_chunks", 5)),
            min_related_chunks=kwargs.get("min_related_chunks", query_config.get("min_related_chunks", 1)),
            enable_candidate_filtering=kwargs.get(
                "enable_candidate_filtering", query_config.get("enable_candidate_filtering", True)
            ),
            candidate_top_k=kwargs.get("candidate_top_k", query_config.get("candidate_top_k", 1000)),
            num_query_expansions=kwargs.get("num_query_expansions", query_config.get("num_query_expansions", 2)),
            min_intersection_size=kwargs.get("min_intersection_size", query_config.get("min_intersection_size", 50)),
            max_tokens_query_expansion=kwargs.get(
                "max_tokens_query_expansion", query_config.get("max_tokens_query_expansion", 1000)
            ),
            high_similarity_threshold=kwargs.get(
                "high_similarity_threshold", query_config.get("high_similarity_threshold", 0.95)
            ),
            enable_rerank=kwargs.get("enable_rerank", self.config.get("rerank", {}).get("enable_by_default", True)),
            min_rerank_score=kwargs.get("min_rerank_score", self.config.get("rerank", {}).get("min_score", 0.5)),
            cosine_threshold=kwargs.get("cosine_threshold", query_config.get("cosine_threshold", 0.2)),
            response_type=kwargs.get(
                "response_type",
                self.config.get("prompts", {}).get("default_response_type", "Multiple Paragraphs"),
            ),
            enable_hybrid_search=kwargs.get(
                "enable_hybrid_search", self.config.get("hybrid_search", {}).get("enabled", False)
            ),
            force_keyword_expansion=kwargs.get("force_keyword_expansion", False),
            log_dir=kwargs.get("log_dir", query_config.get("log_dir")),
            query_id=kwargs.get("query_id"),
            enable_hyde=kwargs.get("enable_hyde", query_config.get("enable_hyde", False)),
            min_hyde_expansions=kwargs.get("min_hyde_expansions", query_config.get("min_hyde_expansions", 3)),
            max_hyde_expansions=kwargs.get("max_hyde_expansions", query_config.get("max_hyde_expansions", 5)),
            max_tokens_hyde=kwargs.get("max_tokens_hyde", query_config.get("max_tokens_hyde", 1500)),
            hyde_temperature=kwargs.get("hyde_temperature", query_config.get("hyde_temperature", 0.5)),
            # Query Decomposition
            enable_query_decomposition=kwargs.get(
                "enable_query_decomposition", query_config.get("enable_query_decomposition", False)
            ),
            max_decomposed_queries=kwargs.get("max_decomposed_queries", query_config.get("max_decomposed_queries", 6)),
            decomposition_temperature=kwargs.get(
                "decomposition_temperature", query_config.get("decomposition_temperature", 0.3)
            ),
            max_tokens_decomposition=kwargs.get(
                "max_tokens_decomposition", query_config.get("max_tokens_decomposition", 4000)
            ),
            # Component control flags
            enable_entity_retrieval=kwargs.get(
                "enable_entity_retrieval", query_config.get("enable_entity_retrieval", True)
            ),
            enable_vector_retrieval=kwargs.get(
                "enable_vector_retrieval", query_config.get("enable_vector_retrieval", True)
            ),
            enable_keyword_retrieval=kwargs.get(
                "enable_keyword_retrieval", query_config.get("enable_keyword_retrieval", True)
            ),
        )

        # Build global config for query functions
        global_config = {
            "tokenizer": None,  # Will use default tokenizer
            "rerank_processor": self.rerank_processor,
            "min_rerank_score": param.min_rerank_score,  # Pass min_rerank_score from param
            "rerank_entity": self.config.get("rerank_entity", {}),
            "rerank_chunk": self.config.get("rerank_chunk", {}),
            "ngram_entity_matching": self.config.get("ngram_entity_matching", {}),
            "second_stage_entity_discovery": self.config.get("second_stage_entity_discovery", {}),
            "entity_stats_manager": self.entity_stats_manager,
            "hybrid_search": self.config.get("hybrid_search", {}),
            "query": query_config,  # CRITICAL: Include full query section for Neo4j expansion and other settings
            "ENABLE_CANDIDATE_FILTERING": query_config.get("enable_candidate_filtering", True),
            "NUM_QUERY_EXPANSIONS": query_config.get("num_query_expansions", 5),
            "MAX_TOKENS_QUERY_EXPANSION": query_config.get("max_tokens_query_expansion", 5000),
        }

        # Build storage dictionary for KG modes
        storage_dict = {
            "chunks_vdb": self.chunks_vdb,
            "entities_vdb": self.entities_vdb,
            "relationships_vdb": self.relationships_vdb,
            "text_chunks_storage": self.text_chunks_storage,
            "chunk_entity_relation_storage": self.chunk_entity_relation_storage,
            "graph_storage": self.graph_storage,
        }

        # Route to appropriate query function
        if mode == "naive":
            return await naive_query(
                query=query_text,
                embedding_manager=self.embedding_manager,
                chunks_vdb=self.chunks_vdb,
                text_chunks_storage=self.text_chunks_storage,
                llm_provider=active_llm_provider,
                config=global_config,
                param=param,
                es_client=self.es_client,
                cache_manager=self.cache_manager,
            )
        elif mode == "bypass":
            return await bypass_query(
                query=query_text,
                llm_provider=active_llm_provider,
                config=global_config,
                param=param,
            )
        elif mode == "entities_only":
            # Entities-only mode: Extract entities without retrieving papers
            # Note: Requires entity extraction components to be initialized
            return await entities_only_query(
                query=query_text,
                llm_provider=active_llm_provider,
                config=global_config,
                ngram_matcher=global_config.get("_ngram_matcher"),
                semantic_community=global_config.get("_semantic_community"),
                redis_client=self.text_chunks_storage,
                param=param,
            )
        elif mode == "auto":
            # Auto mode: Use LLM function calling to determine retrieval strategy
            return await query_with_tools(
                query=query_text,
                llm_provider=active_llm_provider,
                config=global_config,
                embedding_manager=self.embedding_manager,
                storage_dict=storage_dict,
                param=param,
                es_client=self.es_client,
                cache_manager=self.cache_manager,
                worker_pool=self.worker_pool,
                rerank_processor=self.rerank_processor,
                detailed_logger=None,
            )
        else:
            # KG modes: local, global, hybrid, mix
            return await kg_query(
                query=query_text,
                embedding_manager=self.embedding_manager,
                storage_dict=storage_dict,
                llm_provider=active_llm_provider,
                config=global_config,
                param=param,
                es_client=self.es_client,
                cache_manager=self.cache_manager,
                worker_pool=self.worker_pool,
            )


# Convenience function for single query
async def query(
    query_text: str,
    mode: str | None = None,
    config_path: str = "config/rag.conf",
    **kwargs,
) -> dict[str, Any]:
    """
    Convenience function for single query.

    Creates app, initializes, queries, and closes.

    Args:
        query_text: User's query
        mode: Query mode (naive, local, global, hybrid, mix, bypass)
        config_path: Path to configuration file
        **kwargs: Additional query parameters

    Returns:
        Dictionary with response and metadata
    """
    app = RAGApp(config_path=config_path)
    try:
        await app.initialize()
        return await app.query(query_text, mode=mode, **kwargs)
    finally:
        await app.close()
