"""
Query parameters for RAG system.

Based on LightRAG QueryParam.
"""

from dataclasses import dataclass


@dataclass
class QueryParam:
    """
    Query parameters for RAG system.

    Based on LightRAG QueryParam.
    """

    # Query mode
    mode: str = "naive"  # naive, local, global, hybrid, mix, bypass

    # Retrieval parameters
    top_k: int = 10  # Number of entities/relationships
    chunk_top_k: int | None = None  # Number of chunks (None = all until token limit)

    # Token budgets
    max_entity_tokens: int = 2000
    max_relation_tokens: int = 2000
    max_total_tokens: int = 8000

    # Chunk picking
    kg_chunk_pick_method: str = "VECTOR"  # WEIGHT or VECTOR (VECTOR recommended)
    kg_chunk_top_k: int = 50  # Max chunks from entity/relationship sources
    max_related_chunks: int = 5
    min_related_chunks: int = 1

    # Query-guided chunk pre-filtering (performance optimization)
    enable_candidate_filtering: bool = True  # Use LLM query expansion + vector search to pre-filter chunks
    candidate_top_k: int = 1000  # Candidate pool size for vector search
    num_query_expansions: int = 2  # Number of LLM-generated query expansions (legacy fixed mode)
    min_query_expansions: int | None = None  # Minimum query expansions (flexible mode)
    max_query_expansions: int | None = None  # Maximum query expansions (flexible mode)
    min_intersection_size: int = 50  # Minimum intersection size to use filtering
    max_tokens_query_expansion: int = 1000  # Max tokens for query expansion LLM (JSON mode overhead)
    high_similarity_threshold: float = 0.95  # Chunks with similarity >= this are always included

    # Reranking
    enable_rerank: bool = True
    min_rerank_score: float = 0.5

    # Cosine threshold
    cosine_threshold: float = 0.2

    # Response format
    response_type: str = "Multiple Paragraphs"

    # Keyword extraction (LightRAG compatibility)
    hl_keywords: list[str] | None = None  # High-level keywords
    ll_keywords: list[str] | None = None  # Low-level keywords
    only_need_context: bool = False  # Return only context without LLM response

    # Hybrid search (Elasticsearch + Milvus)
    enable_hybrid_search: bool = False  # Enable BM25 + semantic hybrid search
    force_keyword_expansion: bool = False  # Always expand keywords even if cached

    # Detailed logging
    log_dir: str | None = None  # Base directory for detailed logs
    query_id: str | None = None  # Unique query identifier

    # HyDE (Hypothetical Document Embeddings)
    enable_hyde: bool = False
    min_hyde_expansions: int = 3
    max_hyde_expansions: int = 5
    max_tokens_hyde: int = 4000
    hyde_temperature: float = 0.5

    # Query Decomposition (for reranking)
    enable_query_decomposition: bool = False
    max_decomposed_queries: int = 6
    decomposition_temperature: float = 0.3
    max_tokens_decomposition: int = 4000

    # Component control flags (for mix mode retrieval)
    enable_entity_retrieval: bool = True  # Enable entity-based retrieval (KG worker)
    enable_vector_retrieval: bool = True  # Enable vector search (Milvus worker)
    enable_keyword_retrieval: bool = True  # Enable keyword search (ES worker)
