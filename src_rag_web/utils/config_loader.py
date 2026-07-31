"""
Configuration loader for RAG query system.

Loads settings from config/rag.conf with environment variable overrides.
"""

import configparser
import os
from pathlib import Path
from typing import Any


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load RAG configuration from file.

    Args:
        config_path: Path to config file. If None, uses config/rag.conf

    Returns:
        Dict with all configuration sections

    Environment Variables Override:
        - MILVUS_HOST, MILVUS_PORT, MILVUS_WORKSPACE
        - REDIS_HOST, REDIS_PORT
        - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
        - LLM_API_KEY, LLM_MODEL, LLM_PROVIDER
        - RERANK_API_KEY, RERANK_PROVIDER
    """
    # Find config file
    if config_path is None:
        # Default to config/rag.conf relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "rag.conf"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Parse config file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Build config dict with environment variable overrides
    cfg = {}

    # Milvus configuration
    cfg["milvus"] = {
        "host": os.getenv("MILVUS_HOST", config.get("milvus", "host")),
        "port": int(os.getenv("MILVUS_PORT", config.get("milvus", "port"))),
        "workspace": os.getenv("MILVUS_WORKSPACE", config.get("milvus", "workspace")),
        "timeout": int(config.get("milvus", "timeout", fallback="30")),
        "max_retries": int(config.get("milvus", "max_retries", fallback="3")),
    }

    # Redis configuration
    cfg["redis"] = {
        "host": os.getenv("REDIS_HOST", config.get("redis", "host")),
        "port": int(os.getenv("REDIS_PORT", config.get("redis", "port"))),
        "db": int(config.get("redis", "db", fallback="0")),
        "timeout": int(config.get("redis", "timeout", fallback="10")),
        "max_connections": int(config.get("redis", "max_connections", fallback="50")),
    }

    # Neo4j configuration
    cfg["neo4j"] = {
        "uri": os.getenv("NEO4J_URI", config.get("neo4j", "uri")),
        "username": os.getenv("NEO4J_USERNAME", config.get("neo4j", "username")),
        "password": os.getenv("NEO4J_PASSWORD", config.get("neo4j", "password")),
        "max_connection_pool_size": int(config.get("neo4j", "max_connection_pool_size", fallback="100")),
        "connection_timeout": float(config.get("neo4j", "connection_timeout", fallback="30.0")),
        "database": config.get("neo4j", "database", fallback=""),
    }

    # Embedding configuration
    cfg["embedding"] = {
        "chunk_model": config.get("embedding", "chunk_model"),
        "chunk_model_device": config.get("embedding", "chunk_model_device"),
        "chunk_model_max_length": int(config.get("embedding", "chunk_model_max_length")),
        "label_model": config.get("embedding", "label_model"),
        "label_model_device": config.get("embedding", "label_model_device"),
        "label_model_max_length": int(config.get("embedding", "label_model_max_length")),
        "stage2_model": config.get("embedding", "stage2_model", fallback="BAAI/bge-m3"),
        "stage2_model_device": config.get("embedding", "stage2_model_device", fallback="cuda:0"),
        "stage2_model_max_length": int(config.get("embedding", "stage2_model_max_length", fallback="2048")),
        "embedding_dim": int(config.get("embedding", "embedding_dim")),
        "batch_size": int(config.get("embedding", "batch_size", fallback="32")),
    }

    # Query parameters
    cfg["query"] = {
        "default_mode": config.get("query", "default_mode", fallback="mix"),
        "top_k": int(config.get("query", "top_k", fallback="10")),
        "chunk_top_k": int(config.get("query", "chunk_top_k", fallback="10")),
        "max_entity_tokens": int(config.get("query", "max_entity_tokens", fallback="2000")),
        "max_relation_tokens": int(config.get("query", "max_relation_tokens", fallback="2000")),
        "max_total_tokens": int(config.get("query", "max_total_tokens", fallback="8000")),
        "kg_chunk_pick_method": config.get("query", "kg_chunk_pick_method", fallback="WEIGHT"),
        "max_related_chunks": int(config.get("query", "max_related_chunks", fallback="5")),
        "min_related_chunks": int(config.get("query", "min_related_chunks", fallback="1")),
        "cosine_threshold": float(config.get("query", "cosine_threshold", fallback="0.2")),
    }

    # LLM configuration
    cfg["llm"] = {
        "provider": os.getenv("LLM_PROVIDER", config.get("llm", "provider")),
        "model": os.getenv("LLM_MODEL", config.get("llm", "model")),
        "api_key": os.getenv("LLM_API_KEY", config.get("llm", "api_key")),
        "base_url": config.get("llm", "base_url", fallback=""),
        "temperature": float(config.get("llm", "temperature", fallback="0.0")),
        "max_tokens": int(config.get("llm", "max_tokens", fallback="4000")),
        "top_p": float(config.get("llm", "top_p", fallback="1.0")),
        "timeout": int(config.get("llm", "timeout", fallback="60")),
        "enable_cot": config.getboolean("llm", "enable_cot", fallback=True),
        "enable_streaming": config.getboolean("llm", "enable_streaming", fallback=False),
    }

    # Reranking configuration
    cfg["rerank"] = {
        "enable_by_default": config.getboolean("rerank", "enable_by_default", fallback=True),
        "provider": os.getenv("RERANK_PROVIDER", config.get("rerank", "provider")),
        "model": config.get("rerank", "model"),
        "device": config.get("rerank", "device", fallback="cuda:0"),
        "max_length": int(config.get("rerank", "max_length", fallback="512")),
        "batch_size": int(config.get("rerank", "batch_size", fallback="32")),
        "api_key": os.getenv("RERANK_API_KEY", config.get("rerank", "api_key", fallback="")),
        "base_url": config.get("rerank", "base_url", fallback=""),
        "min_score": float(config.get("rerank", "min_score", fallback="0.5")),
        "max_documents": int(config.get("rerank", "max_documents", fallback="100")),
    }

    # Caching configuration
    cfg["cache"] = {
        "enable_llm_cache": config.getboolean("cache", "enable_llm_cache", fallback=True),
        "enable_query_cache": config.getboolean("cache", "enable_query_cache", fallback=True),
        "cache_ttl": int(config.get("cache", "cache_ttl", fallback="86400")),
        "cache_prefix": config.get("cache", "cache_prefix", fallback="rag_cache"),
    }

    # Logging configuration
    cfg["logging"] = {
        "level": config.get("logging", "level", fallback="INFO"),
        "log_file": config.get("logging", "log_file", fallback=""),
        "log_format": config.get(
            "logging", "log_format", fallback="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        "enable_perf_logging": config.getboolean("logging", "enable_perf_logging", fallback=True),
    }

    # Advanced configuration
    cfg["advanced"] = {
        "max_concurrent_ops": int(config.get("advanced", "max_concurrent_ops", fallback="10")),
        "retry_delay": float(config.get("advanced", "retry_delay", fallback="1.0")),
        "enable_profiling": config.getboolean("advanced", "enable_profiling", fallback=False),
        "profile_output_dir": config.get("advanced", "profile_output_dir", fallback="./profiles"),
    }

    # Prompts configuration
    cfg["prompts"] = {
        "rag_response_prompt": config.get("prompts", "rag_response_prompt", fallback=""),
        "naive_response_prompt": config.get("prompts", "naive_response_prompt", fallback=""),
        "keywords_extraction_prompt": config.get("prompts", "keywords_extraction_prompt", fallback=""),
        "default_response_type": config.get("prompts", "default_response_type", fallback="Multiple Paragraphs"),
        "default_language": config.get("prompts", "default_language", fallback="English"),
    }

    # N-gram entity matching configuration
    ngram_section = "ngram_entity_matching"
    k_values_str = config.get(ngram_section, "k_values", fallback="1,2,3,4,5")
    k_values = [int(k.strip()) for k in k_values_str.split(",")]
    entity_collections_str = config.get(ngram_section, "entity_collections", fallback="")
    entity_collections = [c.strip() for c in entity_collections_str.split(",") if c.strip()]

    cfg["ngram_entity_matching"] = {
        "enable": config.getboolean(ngram_section, "enable", fallback=True),
        "statistics_file": config.get(ngram_section, "statistics_file", fallback=""),
        "max_pct_paper": float(config.get(ngram_section, "max_pct_paper", fallback="80.0")),
        "max_pct_chunk": float(config.get(ngram_section, "max_pct_chunk", fallback="80.0")),
        "k_values": k_values,
        "min_similarity": float(config.get(ngram_section, "min_similarity", fallback="0.85")),
        "entity_collections": entity_collections,
        "enable_cache": config.getboolean(ngram_section, "enable_cache", fallback=True),
        "max_cache_size": int(config.get(ngram_section, "max_cache_size", fallback="10000")),
    }

    # Second-stage semantic community entity discovery configuration
    stage2_section = "second_stage_entity_discovery"
    cfg["second_stage_entity_discovery"] = {
        "enable": config.getboolean(stage2_section, "enable", fallback=False),
        "num_shuffles": int(config.get(stage2_section, "num_shuffles", fallback="1000")),
        "appearance_threshold": float(config.get(stage2_section, "appearance_threshold", fallback="0.5")),
        "min_similarity": float(config.get(stage2_section, "min_similarity", fallback="0.85")),
        "top_k": int(config.get(stage2_section, "top_k", fallback="10")),
        "min_entities": int(config.get(stage2_section, "min_entities", fallback="2")),
        "embedding_batch_size": int(config.get(stage2_section, "embedding_batch_size", fallback="200")),
        "query_batch_size": int(config.get(stage2_section, "query_batch_size", fallback="100")),
        "shuffle_strategy": config.get(stage2_section, "shuffle_strategy", fallback="adaptive"),
        "max_exhaustive_snippets": int(config.get(stage2_section, "max_exhaustive_snippets", fallback="6")),
        "min_unique_ratio": float(config.get(stage2_section, "min_unique_ratio", fallback="0.8")),
    }

    return cfg
