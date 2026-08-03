# RAG Query System - Storage Layer

This module implements the storage layer for the RAG query system, extracted from LightRAG.

## Overview

The `src_rag_web` module provides a clean, modular interface to query the knowledge graph database built by LightRAG. It separates query operations from insertion operations, making it easier to maintain and extend — this package is **query-only** and performs no writes to any store.

## Architecture

### Database Triple

The system uses three complementary databases:

1. **Milvus** (Vector Database)
   - Stores entity and chunk embeddings
   - Enables semantic similarity search
   - Collections: `entities`, `chunks`, `relationships`

2. **Redis** (Key-Value Store)
   - Fast content retrieval
   - Entity-chunk mapping
   - Namespaces: `text_chunks`, `entity_sources`, `chunk_entities`, `doc_status`

3. **Neo4j** (Graph Database)
   - Knowledge graph structure
   - Relationship traversal
   - Node labels: `{workspace}`, entity types (Gene, Disease, etc.)
   - Edge type: `DIRECTED` (undirected, stored once)

### Design Principles

1. **Query-Only Operations**: No insert/update/delete operations
2. **Async/Await Pattern**: All database operations are asynchronous
3. **Configuration-Driven**: Settings from `config/rag.conf` with environment variable overrides
4. **Workspace Isolation**: All collections/namespaces prefixed with workspace name
5. **Comprehensive Error Handling**: Logging and error propagation

## Directory Structure

```
src_rag/
├── storage/
│   ├── base.py                 # Abstract base classes
│   ├── milvus_storage.py       # Milvus vector database adapter
│   ├── redis_storage.py        # Redis key-value storage adapter
│   └── neo4j_storage.py        # Neo4j graph database adapter
├── rerank/
│   ├── __init__.py             # Reranker factory and exports
│   ├── local_reranker.py       # Local cross-encoder models
│   ├── api_rerankers.py        # API rerankers (Jina, Cohere, Aliyun)
│   ├── processor.py            # Rerank application logic
│   └── README.md               # Reranking documentation
├── utils/
│   ├── config_loader.py        # Configuration file loader
│   └── __init__.py
└── README.md                   # This file
```

## Storage Adapters

### Base Interfaces (`storage/base.py`)

Three abstract base classes define the storage contract:

- **`BaseVectorStorage`**: Vector similarity search (Milvus)
  - `query()`: Semantic search with embeddings
  - `get_by_id()`, `get_by_ids()`: Retrieve by ID
  - `get_vectors_by_ids()`: Get embeddings for IDs

- **`BaseKVStorage`**: Key-value operations (Redis)
  - `get_by_id()`, `get_by_ids()`: Retrieve values
  - `get_set_members()`: Get Redis set members
  - `scan_keys()`: Scan for matching keys

- **`BaseGraphStorage`**: Graph operations (Neo4j)
  - `get_node()`, `get_nodes_batch()`: Retrieve nodes
  - `get_neighbors()`: Graph traversal
  - `node_degrees_batch()`: Degree counting

### Milvus Storage (`storage/milvus_storage.py`)

Vector database adapter for semantic search.

**Features**:
- Collection loading on initialization
- HNSW index parameters (ef=200)
- Cosine similarity metric
- Partition support
- Batch retrieval with pipeline

**Collections**:
- `{workspace}_entities`: Entity embeddings (1024-dim)
- `{workspace}_chunks`: Text chunk embeddings (1024-dim)
- `{workspace}_relationships`: Relationship embeddings (1024-dim)

**Usage**:
```python
milvus = MilvusStorage(
    workspace="lightrag",
    collection_name="entities",
    config=config["milvus"]
)
await milvus.initialize()
results = await milvus.query(query_text, top_k=10, query_embedding=embedding)
```

### Redis Storage (`storage/redis_storage.py`)

Key-value storage adapter for fast content retrieval.

**Features**:
- Connection pooling (max 50 connections)
- Pipeline batching for efficiency
- JSON parsing with fallback
- Set operations for chunk-entity mappings
- Efficient key scanning (limited to prevent hanging)

**Namespaces**:
- `{workspace}_text_chunks`: Chunk content (JSON)
- `{workspace}_entity_sources`: Entity → Chunks mapping (SEP-delimited)
- `{workspace}_chunk_entities`: Chunk → Entities mapping (Redis set)
- `{workspace}_doc_status`: Document metadata (JSON)

**Usage**:
```python
redis = RedisStorage(
    workspace="lightrag",
    namespace="text_chunks",
    config=config["redis"]
)
await redis.initialize()
chunk = await redis.get_by_id("chunk-xxx")
entities = await redis.get_set_members("chunk-xxx")
```

### Neo4j Storage (`storage/neo4j_storage.py`)

Graph database adapter for knowledge graph traversal.

**Features**:
- Connection pooling (max 100 connections)
- Async session management
- Batch node retrieval
- Efficient graph traversal
- Degree counting with missing node handling
- Edge retrieval (tuple format for compatibility)
- Edge degree calculation (combined node degrees)
- Edge property retrieval

**Schema**:
- Node Labels: `{workspace}` (primary) + entity type
- Node Properties: `entity_id`, `entity_type`, `description`
- Relationship Type: `DIRECTED` (undirected)
- Relationship Properties: `weight`, `description`, `keywords`

**Usage**:
```python
neo4j = Neo4jStorage(
    workspace="lightrag",
    config=config["neo4j"]
)
await neo4j.initialize()

# Node operations
node = await neo4j.get_node("GENE:BRCA1")
nodes = await neo4j.get_nodes_batch(["GENE:BRCA1", "GENE:TP53"])
degree = await neo4j.node_degree("GENE:BRCA1")
degrees = await neo4j.node_degrees_batch(["GENE:BRCA1", "GENE:TP53"])

# Edge operations (tuple format)
edges = await neo4j.get_node_edges("GENE:BRCA1")
# Returns: [("GENE:BRCA1", "GO:0006281"), ("GENE:BRCA1", "GENE:TP53"), ...]

edges_dict = await neo4j.get_nodes_edges_batch(["GENE:BRCA1", "GENE:TP53"])
# Returns: {"GENE:BRCA1": [(...), ...], "GENE:TP53": [(...), ...]}

# Edge properties and degrees
edge = await neo4j.get_edge("GENE:BRCA1", "GO:0006281")
edge_degree = await neo4j.edge_degree("GENE:BRCA1", "GO:0006281")
edge_degrees = await neo4j.edge_degrees_batch([("GENE:BRCA1", "GO:0006281"), ...])

# Neighbor relationships (dict format with properties)
neighbors = await neo4j.get_neighbors("GENE:BRCA1", limit=100)
# Returns: [{"src_id": "GENE:BRCA1", "tgt_id": "GO:0006281", "weight": 10.0, ...}, ...]
```

## Reranking System

The reranking module improves retrieval quality by re-scoring documents using cross-encoder models.

### Features

- **Local Models**: Run HuggingFace cross-encoders on GPU/CPU
- **API Support**: Jina AI, Cohere, Aliyun DashScope
- **High Performance**: ~10-15 docs/sec on GPU with batching
- **Flexible**: Supports custom models and parameters

### Usage

```python
from src_rag.utils.config_loader import load_config
from src_rag.rerank import get_rerank_function, apply_rerank

# Load configuration
config = load_config()
rerank_func = get_rerank_function(config["rerank"])

# Rerank documents
query = "What is machine learning?"
documents = ["ML is AI...", "Neural networks...", "Python programming..."]

results = await rerank_func(query, documents, top_k=2)
# Returns: [(0, 0.95), (1, 0.87)] - (index, relevance_score)

# With document dicts
retrieved_docs = [
    {"id": "doc1", "content": "ML is AI..."},
    {"id": "doc2", "content": "Neural networks..."},
]

reranked_docs = await apply_rerank(
    query=query,
    retrieved_docs=retrieved_docs,
    rerank_func=rerank_func,
    top_n=2,
    min_score=0.5
)
# Each doc now has 'rerank_score' field
```

### Providers

**Local (Recommended)**:
```ini
[rerank]
provider = local
model = BAAI/bge-reranker-v2-m3  # Multilingual
device = cuda:0
max_length = 512
batch_size = 32
```

**Jina AI**:
```ini
provider = jina
model = jina-reranker-v2-base-multilingual
api_key = your-api-key
```

**Cohere**:
```ini
provider = cohere
model = rerank-multilingual-v3.0
api_key = your-api-key
```

See `src_rag/rerank/README.md` for detailed documentation.

## Configuration

Configuration is loaded from `config/rag.conf` using `src_rag/utils/config_loader.py`.

### Configuration Sections

1. **Database Connections** (`milvus`, `redis`, `neo4j`)
2. **Embedding Models** (`embedding`)
3. **Query Parameters** (`query`)
4. **LLM Configuration** (`llm`)
5. **Reranking** (`rerank`)
6. **Caching** (`cache`)
7. **Logging** (`logging`)
8. **Advanced Settings** (`advanced`)

### Environment Variable Overrides

Endpoints, credentials, GPU placement and file paths are overridden with
`BIOGNOSIA_<SECTION>_<KEY>` environment variables, applied by
`_apply_env_overrides()` in `config.py`. Only variables that are set and
non-empty take effect. See the environment-variable reference in the top-level
`README.md` for the full list, and `.env.example` for a template.

Note that `utils/config_loader.py` is a **legacy** loader with a different,
unprefixed variable scheme (`MILVUS_HOST`, `NEO4J_URI`, …). `RAGApp` does not
use it — it imports `load_config` from `config.py`. Do not configure against
those names.

### Loading Configuration

```python
from src_rag_web.config import load_config

config = load_config("config/rag-web.conf")

# Access configuration
workspace = config["milvus"]["workspace"]
host = config["redis"]["host"]
```

## Testing

The test suite lives in the top-level `tests/` directory and runs with pytest.
From the repository root:

```bash
pip install -r testing-env/requirements-dev.txt
pytest
```

The tests are deliberately hermetic — they set `MCP_TEST_MODE=1` and exercise
the MCP protocol surface, the config/env override logic and the citation
footnote resolver without needing torch, a GPU or any database.

## Code References

All code is adapted from LightRAG with the following changes:

1. **Extracted from**:
   - `src/LightRAG/lightrag/kg/milvus_impl.py`
   - `src/LightRAG/lightrag/kg/redis_impl.py`
   - `src/LightRAG/lightrag/kg/neo4j_impl.py`

2. **Key Differences**:
   - Query-only operations (no insert/update)
   - Separate configuration system
   - Cleaner module organization
   - Enhanced error handling and logging
   - Comprehensive test coverage

3. **Compatibility**:
   - Uses same database schema as LightRAG
   - Same ID generation patterns (MD5 hashing)
   - Same workspace isolation
   - Same collection/namespace naming

## Dependencies

Required packages (from `requirements.txt`):

```
# Vector database
pymilvus>=2.3.0

# Key-value storage
redis>=5.0.0

# Graph database
neo4j>=5.0.0

# Utilities
configparser  # Built-in
```

## Related Documentation

- **`../README.md`** — what the system is, the retrieval/rerank/citation
  pipeline, requirements, the two ways to run it, and the full environment
  variable reference.
- **`../config/rag-web.conf`** — every tunable, with inline commentary.

## License

MIT — see `../LICENSE`. Portions of this code are adapted from
[LightRAG](https://github.com/HKUDS/LightRAG) (MIT); see `../NOTICE` for the
full attribution, and for the licences of the pre-trained models this code
downloads and runs.
