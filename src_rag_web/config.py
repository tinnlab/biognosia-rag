"""
Configuration management for RAG system.

Loads and parses configuration from rag.conf file.
"""

import logging
import os
import re
from configparser import ConfigParser
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/rag.conf") -> dict:
    """
    Load configuration from rag.conf file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with all configuration sections
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    parser = ConfigParser()
    parser.read(config_file)

    config = {
        "milvus": parse_milvus_config(parser),
        "redis": parse_redis_config(parser),
        "neo4j": parse_neo4j_config(parser),
        "mongodb": parse_mongodb_config(parser),
        "elasticsearch": parse_elasticsearch_config(parser),
        "embedding": parse_embedding_config(parser),
        "query": parse_query_config(parser),
        "llm": parse_llm_config(parser),
        "rerank": parse_rerank_config(parser),
        "rerank_two_stage": parse_rerank_two_stage_config(parser),
        "rerank_entity": parse_rerank_entity_config(parser),
        "rerank_chunk": parse_rerank_chunk_config(parser),
        "cache": parse_cache_config(parser),
        "logging": parse_logging_config(parser),
        "advanced": parse_advanced_config(parser),
        "prompts": parse_prompts_config(parser),
        "ngram_entity_matching": parse_ngram_config(parser),
        "second_stage_entity_discovery": parse_stage2_config(parser),
        "hybrid_search": parse_hybrid_search_config(parser),
    }

    config = _apply_env_overrides(config)

    return config


def _env_str(config: dict, section: str, key: str, env: str) -> None:
    """Override config[section][key] with os.environ[env] if the var is set (non-empty)."""
    val = os.environ.get(env)
    if val is not None and val != "" and section in config:
        config[section][key] = val


def _env_int(config: dict, section: str, key: str, env: str) -> None:
    val = os.environ.get(env)
    if val is not None and val != "" and section in config:
        try:
            config[section][key] = int(val)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r", env, val)


def _env_float(config: dict, section: str, key: str, env: str) -> None:
    val = os.environ.get(env)
    if val is not None and val != "" and section in config:
        try:
            config[section][key] = float(val)
        except ValueError:
            logger.warning("Ignoring non-float %s=%r", env, val)


def _env_bool(config: dict, section: str, key: str, env: str) -> None:
    val = os.environ.get(env)
    if val is not None and val != "" and section in config:
        config[section][key] = val.strip().lower() in ("1", "true", "yes", "on")


# `.env.example` marks a value the user must supply as <something-like-this>.
# Nothing legitimate in a model name or an API key looks like that.
#
# The knowledge-base settings all ship filled in — the accounts are read-only
# and published with the paper — so there is no longer a startup guard blocking
# boot on unfilled endpoints or credentials. This pattern is still used to catch
# an unedited LLM API key, which is the one value the user really must provide;
# see validate_llm_config below.
PLACEHOLDER_RE = re.compile(r"<[^<>\s]{3,}>")


KNOWN_LLM_PROVIDERS = ("openai", "groq", "anthropic", "ollama")


def validate_llm_config(llm_cfg: dict) -> None:
    """Reject an unusable LLM configuration before any expensive work starts.

    Only relevant when this process owns its LLM (the plain-Python entry point,
    or the MCP server with MCP_LLM_MODE=standalone). Initialization loads
    several GB of models onto the GPU and starts the rerank worker processes
    before anything touches the LLM, so without this a typo in the model name
    first shows up many minutes later as an opaque provider error on the first
    query.
    """
    provider = (llm_cfg.get("provider") or "").strip().lower()
    model = (llm_cfg.get("model") or "").strip()
    base_url = (llm_cfg.get("base_url") or "").strip()
    api_key = (llm_cfg.get("api_key") or "").strip()

    problems = []
    if provider not in KNOWN_LLM_PROVIDERS:
        problems.append(
            f"unknown provider {provider!r} — set BIOGNOSIA_LLM_PROVIDER to one of "
            f"{', '.join(KNOWN_LLM_PROVIDERS)}"
        )
    if not model or PLACEHOLDER_RE.search(model):
        problems.append("no model — set BIOGNOSIA_LLM_MODEL")
    if api_key and PLACEHOLDER_RE.search(api_key):
        problems.append(
            "BIOGNOSIA_LLM_API_KEY is still the placeholder from .env.example — set your "
            "own key, or clear it if the endpoint needs no auth"
        )
    if (
        provider in ("openai", "groq")
        and not base_url
        and not api_key
        and not os.environ.get("OPENAI_API_KEY")
        and not os.environ.get("LLM_API_KEY")
    ):
        problems.append(
            "neither base_url nor api_key — set BIOGNOSIA_LLM_BASE_URL for a local or "
            "self-hosted endpoint, or BIOGNOSIA_LLM_API_KEY for a hosted API"
        )
    if problems:
        raise RuntimeError("The LLM configuration is unusable: " + "; ".join(problems))

    if provider == "ollama":
        # OllamaProvider speaks the legacy /api/generate API and has no
        # generate_with_tools, so auto mode cannot route and every query
        # degrades to full retrieval.
        logger.warning(
            "provider=ollama uses the legacy /api/generate API and cannot do function "
            "calling; prefer BIOGNOSIA_LLM_PROVIDER=openai with "
            "BIOGNOSIA_LLM_BASE_URL=http://<host>:11434/v1"
        )


def require_llm_sdk(provider: str) -> None:
    """Fail early, not on the first query, if the client library is absent."""
    package = {"openai": "openai", "groq": "openai", "anthropic": "anthropic"}.get(
        (provider or "").strip().lower()
    )
    if package is None:
        return
    try:
        __import__(package)
    except ImportError:
        raise RuntimeError(
            f"LLM provider {provider!r} requires the '{package}' package "
            f"(pip install {package})."
        ) from None


def warn_if_hf_token_missing() -> None:
    """Warn — do not fail — when no usable HuggingFace token is configured.

    The four models this system loads are not gated, so an anonymous download
    does work and a machine with a warm cache needs no token at all. But the
    reranker starts many worker processes that each contact the Hub, and
    anonymous requests are rate-limited; an HTTP 429 there surfaces as a
    startup failure. Hence a loud warning rather than a hard error.
    """
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token or PLACEHOLDER_RE.search(token):
        logger.warning(
            "HF_TOKEN is not set. Model downloads will be anonymous and subject to "
            "HuggingFace rate limits, which can fail model loading with HTTP 429. "
            "Set HF_TOKEN in .env to a free HuggingFace access token."
        )


def _apply_env_overrides(config: dict) -> dict:
    """
    Override DB endpoints, credentials, GPU devices and file paths from
    environment variables so a single image can point at different hosts per
    environment (app-only deployment against external databases).

    Only variables that are actually set (non-empty) take effect; otherwise the
    values parsed from the config file are kept.
    """
    # ── MongoDB ──────────────────────────────────────────────────────────────
    _env_str(config, "mongodb", "uri", "BIOGNOSIA_MONGODB_URI")
    _env_str(config, "mongodb", "database", "BIOGNOSIA_MONGODB_DATABASE")
    _env_str(config, "mongodb", "collection", "BIOGNOSIA_MONGODB_COLLECTION")

    # ── Milvus ───────────────────────────────────────────────────────────────
    _env_str(config, "milvus", "host", "BIOGNOSIA_MILVUS_HOST")
    _env_int(config, "milvus", "port", "BIOGNOSIA_MILVUS_PORT")
    _env_str(config, "milvus", "workspace", "BIOGNOSIA_MILVUS_WORKSPACE")

    # ── Redis ────────────────────────────────────────────────────────────────
    _env_str(config, "redis", "host", "BIOGNOSIA_REDIS_HOST")
    _env_int(config, "redis", "port", "BIOGNOSIA_REDIS_PORT")
    _env_int(config, "redis", "db", "BIOGNOSIA_REDIS_DB")
    _env_str(config, "redis", "password", "BIOGNOSIA_REDIS_PASSWORD")

    # ── Neo4j ────────────────────────────────────────────────────────────────
    _env_str(config, "neo4j", "uri", "BIOGNOSIA_NEO4J_URI")
    _env_str(config, "neo4j", "username", "BIOGNOSIA_NEO4J_USERNAME")
    _env_str(config, "neo4j", "password", "BIOGNOSIA_NEO4J_PASSWORD")
    _env_str(config, "neo4j", "database", "BIOGNOSIA_NEO4J_DATABASE")

    # ── Elasticsearch ────────────────────────────────────────────────────────
    _env_str(config, "elasticsearch", "hosts", "BIOGNOSIA_ELASTICSEARCH_HOSTS")
    _env_str(config, "elasticsearch", "username", "BIOGNOSIA_ELASTICSEARCH_USERNAME")
    _env_str(config, "elasticsearch", "password", "BIOGNOSIA_ELASTICSEARCH_PASSWORD")
    _env_bool(config, "elasticsearch", "enabled", "BIOGNOSIA_ELASTICSEARCH_ENABLED")

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Used whenever this process owns its LLM: the plain-Python entry point
    # (examples/query.py) and the MCP server with MCP_LLM_MODE=standalone. In
    # MCP sampling mode the app-level provider is never built and these are
    # inert. Credentials belong here (i.e. in .env), never in rag-web.conf.
    _env_str(config, "llm", "provider", "BIOGNOSIA_LLM_PROVIDER")
    _env_str(config, "llm", "model", "BIOGNOSIA_LLM_MODEL")
    _env_str(config, "llm", "base_url", "BIOGNOSIA_LLM_BASE_URL")
    _env_str(config, "llm", "api_key", "BIOGNOSIA_LLM_API_KEY")
    _env_int(config, "llm", "timeout", "BIOGNOSIA_LLM_TIMEOUT")
    _env_bool(config, "llm", "enable_cot", "BIOGNOSIA_LLM_ENABLE_COT")
    _env_float(config, "llm", "temperature", "BIOGNOSIA_LLM_TEMPERATURE")
    _env_float(config, "llm", "top_p", "BIOGNOSIA_LLM_TOP_P")
    _env_int(config, "llm", "max_tokens", "BIOGNOSIA_LLM_MAX_TOKENS")
    _env_int(config, "llm", "max_completion_tokens", "BIOGNOSIA_LLM_MAX_COMPLETION_TOKENS")
    # The provider always prefers max_completion_tokens when both are present,
    # so drop the one that would be silently ignored rather than carry a value
    # that looks effective but is not.
    if "max_completion_tokens" in config.get("llm", {}):
        config["llm"].pop("max_tokens", None)

    # ── GPU devices (embedding + rerank) ─────────────────────────────────────
    # BIOGNOSIA_EMBEDDING_DEVICE is a convenience that sets all three embedding
    # devices unless a more specific override is also provided.
    shared_embed = os.environ.get("BIOGNOSIA_EMBEDDING_DEVICE")
    if shared_embed and "embedding" in config:
        for k in ("chunk_model_device", "label_model_device", "stage2_model_device"):
            config["embedding"][k] = shared_embed
    _env_str(config, "embedding", "chunk_model_device", "BIOGNOSIA_EMBEDDING_CHUNK_DEVICE")
    _env_str(config, "embedding", "label_model_device", "BIOGNOSIA_EMBEDDING_LABEL_DEVICE")
    _env_str(config, "embedding", "stage2_model_device", "BIOGNOSIA_EMBEDDING_STAGE2_DEVICE")
    _env_str(config, "rerank", "device", "BIOGNOSIA_RERANK_DEVICE")
    _env_str(config, "rerank_two_stage", "stage1_device", "BIOGNOSIA_RERANK_STAGE1_DEVICE")

    # ── Rerank worker-pool sizing (VRAM) ─────────────────────────────────────
    # Each rerank worker is a SEPARATE process that loads the model with its own
    # CUDA context, so these counts drive VRAM directly (measured on an H200:
    # ~1.1 GB per stage-2 worker, ~0.5 GB per stage-1 worker, on top of ~3.7 GB
    # for the main process). A deployment sharing a GPU with other workloads has
    # to size these WITHOUT rebuilding the image or bind-mounting a whole edited
    # copy of rag-web.conf, which was previously the only option.
    _env_int(config, "rerank", "num_workers", "BIOGNOSIA_RERANK_NUM_WORKERS")
    _env_int(
        config,
        "rerank_two_stage",
        "stage1_num_workers",
        "BIOGNOSIA_RERANK_STAGE1_NUM_WORKERS",
    )

    # ── File paths ───────────────────────────────────────────────────────────
    _env_str(config, "query", "log_dir", "BIOGNOSIA_QUERY_LOG_DIR")
    _env_str(config, "ngram_entity_matching", "statistics_file", "BIOGNOSIA_NGRAM_STATISTICS_FILE")

    return config


def parse_milvus_config(parser: ConfigParser) -> dict:
    """Parse Milvus configuration."""
    if not parser.has_section("milvus"):
        return {}

    section = parser["milvus"]
    return {
        "host": section.get("host", "localhost"),
        "port": section.getint("port", 19530),
        "workspace": section.get("workspace", "lightrag"),
        "timeout": section.getint("timeout", 30),
        "max_retries": section.getint("max_retries", 3),
        "entity_search_max_workers": section.getint("entity_search_max_workers", 10),
    }


def parse_redis_config(parser: ConfigParser) -> dict:
    """Parse Redis configuration."""
    if not parser.has_section("redis"):
        return {}

    section = parser["redis"]
    return {
        "host": section.get("host", "localhost"),
        "port": section.getint("port", 6379),
        "db": section.getint("db", 0),
        "password": section.get("password", ""),
        "timeout": section.getint("timeout", 10),
        "max_connections": section.getint("max_connections", 50),
    }


def parse_neo4j_config(parser: ConfigParser) -> dict:
    """Parse Neo4j configuration."""
    if not parser.has_section("neo4j"):
        return {}

    section = parser["neo4j"]
    return {
        "uri": section.get("uri", "bolt://localhost:7687"),
        "username": section.get("username", "neo4j"),
        "password": section.get("password", ""),
        "max_connection_pool_size": section.getint("max_connection_pool_size", 100),
        "connection_timeout": section.getfloat("connection_timeout", 30.0),
        "database": section.get("database", ""),
    }


def parse_mongodb_config(parser: ConfigParser) -> dict:
    """Parse MongoDB configuration."""
    if not parser.has_section("mongodb"):
        return {}

    section = parser["mongodb"]
    return {
        "uri": section.get("uri", "mongodb://localhost:27017"),
        "database": section.get("database", "paper_db"),
        "collection": section.get("collection", "papers"),
    }


def parse_elasticsearch_config(parser: ConfigParser) -> dict:
    """Parse Elasticsearch configuration."""
    if not parser.has_section("elasticsearch"):
        return {"enabled": False}

    section = parser["elasticsearch"]
    return {
        "enabled": section.getboolean("enabled", False),
        "hosts": section.get("hosts", "http://localhost:9200"),
        "username": section.get("username", ""),
        "password": section.get("password", ""),
        "index_name": section.get("index_name", "lightrag_chunks"),
        "number_of_shards": section.getint("number_of_shards", 3),
        "number_of_replicas": section.getint("number_of_replicas", 1),
        "default_fuzziness": section.get("default_fuzziness", "AUTO"),
        "default_operator": section.get("default_operator", "or"),
        "timeout": section.getint("timeout", 30),
        "max_retries": section.getint("max_retries", 3),
    }


def parse_embedding_config(parser: ConfigParser) -> dict:
    """Parse embedding configuration."""
    if not parser.has_section("embedding"):
        return {}

    section = parser["embedding"]
    return {
        "chunk_model": section.get("chunk_model", "BAAI/bge-m3"),
        "chunk_model_device": section.get("chunk_model_device", "cuda:0"),
        "chunk_model_max_length": section.getint("chunk_model_max_length", 512),
        "label_model": section.get("label_model", "ncbi/MedCPT-Query-Encoder"),
        "label_model_device": section.get("label_model_device", "cuda:0"),
        "label_model_max_length": section.getint("label_model_max_length", 256),
        "stage2_model": section.get("stage2_model", "BAAI/bge-m3"),
        "stage2_model_device": section.get("stage2_model_device", "cuda:0"),
        "stage2_model_max_length": section.getint("stage2_model_max_length", 2048),
        "embedding_dim": section.getint("embedding_dim", 1024),
        "batch_size": section.getint("batch_size", 32),
    }


def parse_query_config(parser: ConfigParser) -> dict:
    """Parse query configuration."""
    if not parser.has_section("query"):
        return {}

    section = parser["query"]
    return {
        "default_mode": section.get("default_mode", "mix"),
        "top_k": section.getint("top_k", 10),
        "chunk_top_k": section.getint("chunk_top_k", 10),
        "max_entity_tokens": section.getint("max_entity_tokens", 2000),
        "max_relation_tokens": section.getint("max_relation_tokens", 2000),
        "max_total_tokens": section.getint("max_total_tokens", 8000),
        "kg_chunk_pick_method": section.get("kg_chunk_pick_method", "WEIGHT"),
        "kg_chunk_top_k": section.getint("kg_chunk_top_k", 50),
        "max_related_chunks": section.getint("max_related_chunks", 5),
        "min_related_chunks": section.getint("min_related_chunks", 1),
        "cosine_threshold": section.getfloat("cosine_threshold", 0.2),
        # Query-guided chunk pre-filtering optimization parameters
        "enable_candidate_filtering": section.getboolean("enable_candidate_filtering", True),
        "candidate_top_k": section.getint("candidate_top_k", 1000),
        "num_query_expansions": section.getint("num_query_expansions", 2),
        "min_query_expansions": (
            section.getint("min_query_expansions", fallback=None) if section.get("min_query_expansions") else None
        ),
        "max_query_expansions": (
            section.getint("max_query_expansions", fallback=None) if section.get("max_query_expansions") else None
        ),
        "min_intersection_size": section.getint("min_intersection_size", 50),
        "max_tokens_query_expansion": section.getint("max_tokens_query_expansion", 4000),
        "high_similarity_threshold": section.getfloat("high_similarity_threshold", 0.95),
        # Detailed logging
        "log_dir": section.get("log_dir", "").strip() or None,
        # HyDE
        "enable_hyde": section.getboolean("enable_hyde", False),
        "min_hyde_expansions": section.getint("min_hyde_expansions", 3),
        "max_hyde_expansions": section.getint("max_hyde_expansions", 5),
        "max_tokens_hyde": section.getint("max_tokens_hyde", 4000),
        "hyde_temperature": section.getfloat("hyde_temperature", 0.5),
        # Query Decomposition
        "enable_query_decomposition": section.getboolean("enable_query_decomposition", False),
        "max_decomposed_queries": section.getint("max_decomposed_queries", 6),
        "decomposition_temperature": section.getfloat("decomposition_temperature", 0.3),
        "max_tokens_decomposition": section.getint("max_tokens_decomposition", 4000),
        # Neo4j entity expansion
        "neo4j_entity_expansion_hops": section.getint("neo4j_entity_expansion_hops", 1),
        "neo4j_expansion_min_weight_percentile": section.getfloat("neo4j_expansion_min_weight_percentile", 0.1),
        "neo4j_expansion_max_weight_percentile": section.getfloat("neo4j_expansion_max_weight_percentile", 0.95),
        "neo4j_expansion_max_per_hop": section.getint("neo4j_expansion_max_per_hop", 10),
        "neo4j_expansion_relationship_types": section.get(
            "neo4j_expansion_relationship_types", "REGULATES,INTERACTS_WITH,PART_OF"
        ),
        "max_entity_description_length": section.getint("max_entity_description_length", 300),
        # Component control flags for mix mode
        "enable_entity_retrieval": section.getboolean("enable_entity_retrieval", True),
        "enable_vector_retrieval": section.getboolean("enable_vector_retrieval", True),
        "enable_keyword_retrieval": section.getboolean("enable_keyword_retrieval", True),
    }


def parse_llm_config(parser: ConfigParser) -> dict:
    """Parse LLM configuration.

    Returns a usable dict even when the file has no [llm] section. The committed
    config deliberately ships that section commented out — an LLM endpoint and
    key are per-user settings and a key must never be committed — so in normal
    use the whole section arrives from BIOGNOSIA_LLM_* environment variables
    (see _apply_env_overrides). Returning defaults unconditionally keeps the
    shape of config["llm"] stable regardless of which of those happen to be set.
    """
    section = parser["llm"] if parser.has_section("llm") else None

    def _get(key, default):
        return section.get(key, default) if section is not None else default

    def _getint(key, default):
        return section.getint(key, default) if section is not None else default

    def _getbool(key, default):
        return section.getboolean(key, default) if section is not None else default

    # "openai" is the right default for ANY OpenAI-compatible endpoint. Do not
    # change it to "groq": supports_json_mode() returns True unconditionally for
    # groq, which makes the pipeline request response_format={"type":
    # "json_object"} from endpoints whose backend may not support it.
    config = {
        "provider": _get("provider", "openai"),
        "model": _get("model", ""),
        "api_key": _get("api_key", ""),
        "base_url": _get("base_url", ""),
        "timeout": _getint("timeout", 300),  # generous: reasoning models are slow
        "enable_cot": _getbool("enable_cot", True),
        "enable_streaming": _getbool("enable_streaming", False),
    }

    # Only add optional parameters if they are explicitly set. An ABSENT key
    # means "do not send this parameter to the API at all", which is what lets
    # reasoning models that reject temperature/top_p/max_tokens work.
    if section is not None:
        if section.get("temperature") is not None:
            config["temperature"] = section.getfloat("temperature")

        if section.get("top_p") is not None:
            config["top_p"] = section.getfloat("top_p")

        # Prefer max_completion_tokens over max_tokens (for newer OpenAI models)
        if section.get("max_completion_tokens") is not None:
            config["max_completion_tokens"] = section.getint("max_completion_tokens")
        elif section.get("max_tokens") is not None:
            config["max_tokens"] = section.getint("max_tokens")

    return config


def parse_rerank_config(parser: ConfigParser) -> dict:
    """Parse reranking configuration."""
    if not parser.has_section("rerank"):
        return {}

    section = parser["rerank"]
    return {
        "enable_by_default": section.getboolean("enable_by_default", True),
        "provider": section.get("provider", "local"),
        "model": section.get("model", "jinaai/jina-reranker-v2-base-multilingual"),
        "device": section.get("device", "cuda:0"),
        "max_length": section.getint("max_length", 512),
        "batch_size": section.getint("batch_size", 32),
        "num_workers": section.getint("num_workers", 10),
        "api_key": section.get("api_key", ""),
        "base_url": section.get("base_url", ""),
        "min_score": section.getfloat("min_score", 0.5),
        "max_documents": section.getint("max_documents", 100),
        "early_stop_target": section.getint("early_stop_target", 5),
    }


def parse_rerank_two_stage_config(parser: ConfigParser) -> dict:
    """Parse two-stage reranking configuration."""
    if not parser.has_section("rerank.two_stage"):
        return {"enabled": False}

    section = parser["rerank.two_stage"]
    return {
        "enabled": section.getboolean("enabled", False),
        "stage1_model": section.get("stage1_model", "cross-encoder/ms-marco-TinyBERT-L2-v2"),
        "stage1_device": section.get("stage1_device", "cuda:1"),
        "stage1_top_k": section.getint("stage1_top_k", 10000),
        "stage1_min_score": section.getfloat("stage1_min_score", 0.0),
        "stage1_batch_size": section.getint("stage1_batch_size", 4000),
        "stage1_num_workers": section.getint("stage1_num_workers", 10),
        "stage1_max_length": section.getint("stage1_max_length", 512),
        "stage2_min_score": section.getfloat("stage2_min_score", 0.5),
    }


def parse_rerank_entity_config(parser: ConfigParser) -> dict:
    """Parse entity reranking configuration."""
    if not parser.has_section("rerank_entity"):
        return {"enable_stage1": False, "enable_stage2": False}

    section = parser["rerank_entity"]
    return {
        "enable_stage1": section.getboolean("enable_stage1", False),
        "enable_stage2": section.getboolean("enable_stage2", False),
        "content_field": section.get("content_field", "content"),
        "min_score": section.getfloat("min_score", 0.3),
        "max_entities": section.getint("max_entities", 200),
        "fallback": section.get("fallback", "use_name"),
    }


def parse_rerank_chunk_config(parser: ConfigParser) -> dict:
    """Parse chunk reranking configuration for multi-query reranking."""
    if not parser.has_section("rerank_chunk"):
        return {
            "score_aggregation": "max",
        }

    section = parser["rerank_chunk"]
    return {
        "score_aggregation": section.get("score_aggregation", "max"),
    }


def parse_cache_config(parser: ConfigParser) -> dict:
    """Parse caching configuration."""
    if not parser.has_section("cache"):
        return {}

    section = parser["cache"]
    return {
        "enable_llm_cache": section.getboolean("enable_llm_cache", True),
        "enable_query_cache": section.getboolean("enable_query_cache", True),
        "cache_ttl": section.getint("cache_ttl", 86400),
        "cache_prefix": section.get("cache_prefix", "rag_cache"),
    }


def parse_logging_config(parser: ConfigParser) -> dict:
    """Parse logging configuration."""
    if not parser.has_section("logging"):
        return {}

    section = parser["logging"]
    return {
        "level": section.get("level", "INFO"),
        "log_file": section.get("log_file", ""),
        "log_format": section.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        "enable_perf_logging": section.getboolean("enable_perf_logging", True),
    }


def parse_advanced_config(parser: ConfigParser) -> dict:
    """Parse advanced configuration."""
    if not parser.has_section("advanced"):
        return {}

    section = parser["advanced"]
    return {
        "max_concurrent_ops": section.getint("max_concurrent_ops", 10),
        "retry_delay": section.getfloat("retry_delay", 1.0),
        "enable_profiling": section.getboolean("enable_profiling", False),
        "profile_output_dir": section.get("profile_output_dir", "./profiles"),
    }


def parse_prompts_config(parser: ConfigParser) -> dict:
    """Parse prompts configuration."""
    if not parser.has_section("prompts"):
        return {}

    section = parser["prompts"]
    return {
        "rag_response_prompt": section.get("rag_response_prompt", ""),
        "naive_response_prompt": section.get("naive_response_prompt", ""),
        "keywords_extraction_prompt": section.get("keywords_extraction_prompt", ""),
        "default_response_type": section.get("default_response_type", "Multiple Paragraphs"),
        "default_language": section.get("default_language", "English"),
    }


def parse_ngram_config(parser: ConfigParser) -> dict:
    """
    Parse n-gram entity matching configuration.

    Returns:
        Dict with n-gram matching settings
    """
    if not parser.has_section("ngram_entity_matching"):
        return {"enable": False}

    section = parser["ngram_entity_matching"]

    # Parse k_values
    k_values_str = section.get("k_values", "1,2,3,4,5")
    k_values = [int(k.strip()) for k in k_values_str.split(",") if k.strip()]

    # Parse entity collections
    entity_collections_str = section.get("entity_collections", "")
    if entity_collections_str:
        entity_collections = [c.strip() for c in entity_collections_str.split(",") if c.strip()]
    else:
        entity_collections = None  # Use all collections

    # Parse thresholds (can be None)
    max_pct_paper = None
    max_pct_chunk = None
    max_num_text_variations = None

    try:
        max_pct_paper_str = section.get("max_pct_paper", "")
        if max_pct_paper_str:
            max_pct_paper = float(max_pct_paper_str)
    except (ValueError, TypeError):
        pass

    try:
        max_pct_chunk_str = section.get("max_pct_chunk", "")
        if max_pct_chunk_str:
            max_pct_chunk = float(max_pct_chunk_str)
    except (ValueError, TypeError):
        pass

    try:
        max_num_text_variations_str = section.get("max_num_text_variations", "")
        if max_num_text_variations_str:
            max_num_text_variations = int(max_num_text_variations_str)
    except (ValueError, TypeError):
        pass

    return {
        "enable": section.getboolean("enable", False),
        "statistics_file": section.get("statistics_file", ""),
        "max_pct_paper": max_pct_paper,
        "max_pct_chunk": max_pct_chunk,
        "max_num_text_variations": max_num_text_variations,
        "k_values": k_values,
        "min_similarity": section.getfloat("min_similarity", 0.85),
        "entity_collections": entity_collections,
        "enable_cache": section.getboolean("enable_cache", True),
        "max_cache_size": section.getint("max_cache_size", 10000),
    }


def parse_stage2_config(parser: ConfigParser) -> dict:
    """
    Parse second-stage semantic community entity discovery configuration.

    Supports both config file and environment variable for exclude_collections.
    Environment variable EXCLUDE_COLLECTIONS (comma-separated) is appended to config file setting.

    Returns:
        Dict with Stage 2 settings
    """
    if not parser.has_section("second_stage_entity_discovery"):
        return {"enable": False}

    section = parser["second_stage_entity_discovery"]

    # Parse exclude_collections from config file
    exclude_collections_str = section.get("exclude_collections", "Taxonomy")
    if exclude_collections_str:
        exclude_collections = [c.strip() for c in exclude_collections_str.split(",") if c.strip()]
    else:
        exclude_collections = []

    # Append exclude_collections from environment variable
    env_exclude_collections = os.environ.get("EXCLUDE_COLLECTIONS", "")
    if env_exclude_collections:
        env_collections = [c.strip() for c in env_exclude_collections.split(",") if c.strip()]
        # Merge with config file collections (use set to avoid duplicates)
        exclude_collections = list(set(exclude_collections + env_collections))
        logger.info(
            f"Stage 2: Merged exclude_collections from config ({section.get('exclude_collections', '')}) "
            f"and environment ({env_exclude_collections}) -> {', '.join(exclude_collections)}"
        )

    return {
        "enable": section.getboolean("enable", False),
        "num_shuffles": section.getint("num_shuffles", 1000),
        "appearance_threshold": section.getfloat("appearance_threshold", 0.5),
        "min_similarity": section.getfloat("min_similarity", 0.85),
        "top_k": section.getint("top_k", 10),
        "min_entities": section.getint("min_entities", 2),
        "embedding_batch_size": section.getint("embedding_batch_size", 200),
        "query_batch_size": section.getint("query_batch_size", 100),
        "shuffle_strategy": section.get("shuffle_strategy", "adaptive"),
        "max_exhaustive_snippets": section.getint("max_exhaustive_snippets", 6),
        "min_unique_ratio": section.getfloat("min_unique_ratio", 0.8),
        "exclude_collections": exclude_collections,
        "enable_outlier_removal": section.getboolean("enable_outlier_removal", True),
        "outlier_threshold": section.getfloat("outlier_threshold", 0.60),
    }


def parse_hybrid_search_config(parser: ConfigParser) -> dict:
    """Parse hybrid search configuration."""
    if not parser.has_section("hybrid_search"):
        return {"enabled": False}

    section = parser["hybrid_search"]
    return {
        "enabled": section.getboolean("enabled", False),
        "enable_keyword_expansion": section.getboolean("enable_keyword_expansion", True),
        "max_keywords": section.getint("max_keywords", 30),
        "keyword_cache_ttl": section.getint("keyword_cache_ttl", 86400),
        "es_top_k": section.getint("es_top_k", 50),
        "es_top_k_per_query": section.getint("es_top_k_per_query", 5000),
        "use_query_expansions_for_es": section.getboolean("use_query_expansions_for_es", True),
        "milvus_top_k": section.getint("milvus_top_k", 50),
        "merged_top_k": section.getint("merged_top_k", 100),
        "rrf_k": section.getint("rrf_k", 60),
        "fallback_to_semantic_on_es_error": section.getboolean("fallback_to_semantic_on_es_error", True),
    }
