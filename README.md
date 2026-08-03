# Biognosia RAG

Retrieval-augmented generation over a biomedical literature knowledge base.

This repository contains the **retrieval and serving code**: given a question, it
finds relevant passages across a vector index, a knowledge graph and a keyword
index, reranks them, and asks a language model to answer using only those
passages — returning the answer with a numbered citation for every claim, each
resolving to the paper and passage it came from.

It can be used two ways:

- **As a Python library** — construct the app, call `query()`, get an answer.
  Bring your own LLM API key. This is the simplest path and the one to start with.
- **As an MCP server** — the same pipeline behind a Model Context Protocol
  endpoint, so any MCP-capable client can call it as a tool.

The knowledge base itself is **not** part of this repository. See
[Data availability](#data-availability).

---

## Contents

- [Architecture](#architecture)
- [How a query is answered](#how-a-query-is-answered)
- [Requirements](#requirements)
- [Quick start — Python library](#quick-start--python-library)
- [Running as an MCP server](#running-as-an-mcp-server)
- [Configuration reference](#configuration-reference)
- [Choosing a GPU and sizing the rerank pools](#choosing-a-gpu-and-sizing-the-rerank-pools)
- [Tests](#tests)
- [Data availability](#data-availability)
- [Licence](#licence)
- [How to cite](#how-to-cite)

---

## Architecture

Five stores back the retrieval pipeline:

| Store | Role | Required? |
|-------|------|-----------|
| Milvus | dense vector index over passages and entities | yes |
| Redis | passage text, and the entity → passage index | yes |
| Neo4j | biomedical knowledge graph (entities and their relations) | yes |
| Elasticsearch | BM25 / hybrid keyword search | soft — retrieval falls back to the vector and graph paths |
| MongoDB | paper metadata used to render citations | soft — answers still return, but citations degrade to raw passage text |

Two cross-encoder rerankers and three embedding models run locally on a GPU. The
language model is **not** local: you point the system at an OpenAI-compatible or
Anthropic endpoint, or let an MCP client supply generation.

This package performs **no writes** to any store. It is a query and serving
layer over a knowledge base built elsewhere.

## How a query is answered

1. **Routing.** The model is offered two retrieval tools and decides whether the
   question needs full retrieval, entity lookup only, or a direct answer. A
   model without function calling skips this and always retrieves.
2. **Entity extraction.** Biomedical entities in the question are matched
   against the knowledge graph — first by n-gram matching over entity labels
   embedded with MedCPT, falling back to vector search over the entity index.
3. **Graph expansion.** Matched entities are expanded over `REGULATES`,
   `INTERACTS_WITH` and `PART_OF` edges in Neo4j, then reranked and cut to a
   score threshold so expansion does not dilute the context.
4. **Parallel retrieval.** Three workers run concurrently:
   - *graph* — passages linked to the matched entities and relations;
   - *keyword* — query keywords expanded by the model, then BM25 against
     Elasticsearch, fused with reciprocal rank fusion;
   - *vector* — dense search over Milvus, optionally with query expansion,
     HyDE and sub-question decomposition.
5. **Merge.** Results are merged by passage id, with the retrieval sources that
   found each one tracked, and downselected in a rank-balanced way so no single
   source dominates.
6. **Two-stage reranking.** A small cross-encoder
   (`ms-marco-TinyBERT-L2-v2`) filters the candidate pool down; a stronger
   cross-encoder (`jina-reranker-v2`) scores the survivors. Both run as pools of
   separate processes, which is what makes the worker counts a VRAM decision.
7. **Context building.** Surviving passages are split into confidently-relevant
   and possibly-relevant, ordered, and labelled `[C1]`, `[C2]`, … The model
   never sees raw passage identifiers.
8. **Generation.** The model answers, citing the labels it used.
9. **Citation resolution.** Labels are mapped back to passages, paper metadata
   is looked up in MongoDB, and the answer is rewritten with numbered footnotes.
   Passages from the same paper are grouped under one number, and any label the
   model invented is dropped rather than shown.

## Requirements

**Hardware**

- An NVIDIA GPU. The models occupy roughly **10 GB** with the recommended 4 + 4
  rerank workers, but **peak usage while reranking a real query reached about
  84 GB** in our measurement — see
  [sizing](#choosing-a-gpu-and-sizing-the-rerank-pools) before choosing a card,
  and turn the batch settings down if you have less.
- ~10 GB of disk for the model weights, or ~22 GB for the Docker image (which
  bakes them in along with CUDA and torch).
- The knowledge-base endpoints you will query (see
  [Data availability](#data-availability)).

**Software**

- Python 3.11, and CUDA 12.4 drivers.
- For the Docker path: Docker v25 or newer, the NVIDIA Container Toolkit, and a
  CDI spec on the host:
  ```bash
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   # re-run if GPUs change
  sudo docker run --rm --device nvidia.com/gpu=all \
    nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L          # smoke test
  ```
  No Docker daemon restart is needed; Docker v25+ reads the CDI spec directly.

**An LLM endpoint.** Anything OpenAI-compatible (OpenAI, Groq, vLLM, LM Studio,
Ollama's `/v1`) or the Anthropic API. This repository ships no model weights for
generation and makes no assumption about which model you use — though routing
(step 1 above) works best with a model that supports function calling.

**A HuggingFace account and access token.** Strongly recommended rather than
strictly required: none of the four models below is gated, so anonymous download
does work, but the reranker starts many worker processes that each contact the
Hub, and anonymous requests are rate-limited. An HTTP 429 during model loading
becomes a startup failure. A free token raises the limit.

**Models downloaded at run time (Python) or build time (Docker):**

| Model | Role | Licence |
|---|---|---|
| [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | passage and query embeddings | MIT |
| [`ncbi/MedCPT-Query-Encoder`](https://huggingface.co/ncbi/MedCPT-Query-Encoder) | entity-label embeddings | public domain (NLM/NCBI) |
| [`jinaai/jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | second-stage reranking | **CC-BY-NC-4.0 — non-commercial only** |
| [`cross-encoder/ms-marco-TinyBERT-L2-v2`](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2) | first-stage reranking | Apache-2.0 |

> **Licence notice.** The second-stage reranker is licensed for research and
> evaluation only. Nothing in this repository enforces that — the model
> downloads without any agreement step — so complying with it is the
> responsibility of whoever runs this software. For commercial use, substitute a
> differently-licensed reranker via `[rerank]` in `config/rag-web.conf`. All four
> licences are recorded in [`NOTICE`](NOTICE).

## Quick start — Python library

```bash
git clone https://github.com/tinnlab/biognosia-rag.git
cd biognosia-rag

python3.11 -m venv .venv && source .venv/bin/activate

# torch must come from the CUDA 12.4 wheel index, not PyPI.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# NLTK corpora, used by entity extraction and keyword processing. Not a pip
# dependency — without them, n-gram entity matching silently degrades.
python -m nltk.downloader punkt punkt_tab stopwords \
    averaged_perceptron_tagger averaged_perceptron_tagger_eng

cp .env.example .env
```

`examples/query.py` loads `.env` automatically (`python-dotenv` arrives with
`uvicorn[standard]`); if you strip that dependency, export the variables into
your shell instead.

Now edit `.env`. Everything you must change is in the **`FILL THESE IN`** block
at the very top: the knowledge-base endpoints, your HuggingFace token, and your
LLM key. Then choose your LLM a few lines further down:

```dotenv
BIOGNOSIA_LLM_PROVIDER=openai         # any OpenAI-compatible endpoint
BIOGNOSIA_LLM_MODEL=gpt-4o-mini
# BIOGNOSIA_LLM_BASE_URL=http://localhost:11434/v1   # only for non-OpenAI hosts
```

Then ask a question:

```bash
python examples/query.py "What role does BRCA1 play in homologous recombination repair?"
```

The first run downloads about 5 GB of model weights and loads them onto the GPU,
so give it a few minutes; later runs reuse the cache. If any endpoint is still a
placeholder, the run stops immediately and names the variable rather than timing
out later.

`examples/query.py` is deliberately small — the whole integration is:

```python
from src_rag_web.app import RAGApp

app = RAGApp(config_path="config/rag-web.conf", skip_llm=False)
await app.initialize()                       # loads models, connects to the stores
result = await app.query(query_text=question, mode="auto")
print(result["response"])                    # answer, citing passages as [C1], [C2], ...
for ref in result["references"]:             # the passages it was grounded in
    print(ref["handle"], ref["id"])          # -> C1 chunk-<paper hash>-<n>
await app.close()
```

The library returns the answer with the **passage handles** the model cited
(`[C1]`, `[C2]`, …) plus a `references` list mapping each handle to its passage
id; the first 40 hex characters of that id are the paper's identifier. Turning
those into numbered footnotes with full bibliographic entries — looking the
papers up in MongoDB, grouping passages from the same paper under one number,
and dropping any handle the model invented — is done by the MCP layer
(`_resolve_footnotes`). If you want formatted citations rather than handles,
either run the MCP server or call that resolver yourself.

`skip_llm=False` is what makes this process own its LLM provider, built from the
`BIOGNOSIA_LLM_*` settings. Read `examples/query.py` for the runnable version.

## Running as an MCP server

The server exposes one tool, `query_rag`, over the Model Context Protocol at
`POST /mcp`, plus `GET /health`. It has two modes, set with `MCP_LLM_MODE`:

- **`standalone`** — the server calls the LLM endpoint from your `.env` itself.
  No special client is needed; anything that speaks MCP over HTTP can drive it.
- **`sampling`** (the default) — the server generates nothing locally. For every
  completion it asks the *client* via `sampling/createMessage`, so answers come
  from whatever model the client is already using and the server needs no key of
  its own. This requires a sampling-capable MCP client.

Which mode is active is visible in the `initialize` response: `sampling` appears
in the advertised capabilities only when the server will actually call back.

```bash
cp .env.example .env      # fill in as above; set MCP_LLM_MODE

# The build bakes the four models into the image, so pass your token here.
# The running container needs no token.
export HF_TOKEN=hf_...
sudo docker compose up --build -d

curl http://localhost:8081/health     # {"status":"healthy"} once models + stores are up
```

`/health` returns `503 {"status":"initializing"}` until the models are loaded and
the stores are connected, then `200 {"status":"healthy"}`. Expect this to take
several minutes on first start; the container healthcheck allows for it.

To call the tool directly in standalone mode — note that `tools/call` responds as
an SSE stream:

```bash
SID=$(curl -sD- -o /dev/null -X POST http://localhost:8081/mcp \
  -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}' \
  | tr -d '\r' | awk -F': ' '/^mcp-session-id/{print $2}')

curl -N -X POST http://localhost:8081/mcp \
  -H "mcp-session-id: $SID" -H 'content-type: application/json' \
  -H 'accept: text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"query_rag",
       "arguments":{"query":"What role does BRCA1 play in DNA repair?"}}}'
```

Notes:

- **Single worker only.** Per-session state lives in process memory, so the
  server must run with `--workers 1` (the image already does).
- `/mcp` is unauthenticated by default. Set `MCP_AUTH_TOKEN` to require
  `Authorization: Bearer <token>`, and bind to `127.0.0.1` if the host is not
  firewalled. `/health` stays unauthenticated so orchestrators can probe it.
- `run.sh prod` is a thin `docker run` wrapper if you prefer it to compose.
- `MCP_TEST_MODE=1` starts the server with no databases, models or GPU and
  answers only the protocol handshake — useful for checking client wiring.

## Configuration reference

Defaults live in `config/rag-web.conf`, which documents every retrieval and
reranking tunable inline. Environment variables override them, and only
variables that are set and non-empty take effect. Copy `.env.example` to `.env`
as the starting point.

**Knowledge base** — no working defaults; these are the values you must fill in.

| Variable | Maps to |
|---|---|
| `BIOGNOSIA_MILVUS_HOST` / `_PORT` / `_WORKSPACE` | Milvus. `WORKSPACE` is the collection prefix. |
| `BIOGNOSIA_NEO4J_URI` / `_USERNAME` / `_PASSWORD` / `_DATABASE` | Neo4j |
| `BIOGNOSIA_REDIS_HOST` / `_PORT` / `_DB` / `_PASSWORD` | Redis |
| `BIOGNOSIA_MONGODB_URI` / `_DATABASE` / `_COLLECTION` | MongoDB |
| `BIOGNOSIA_ELASTICSEARCH_HOSTS` / `_USERNAME` / `_PASSWORD` / `_ENABLED` | Elasticsearch |

**LLM** — used by the Python library and by `MCP_LLM_MODE=standalone`.

| Variable | Purpose | Default |
|---|---|---|
| `BIOGNOSIA_LLM_PROVIDER` | `openai` \| `groq` \| `anthropic` \| `ollama`. Use `openai` for any OpenAI-compatible endpoint. | `openai` |
| `BIOGNOSIA_LLM_MODEL` | model name | — (required) |
| `BIOGNOSIA_LLM_BASE_URL` | endpoint for non-OpenAI hosts; **must include `/v1`** | — |
| `BIOGNOSIA_LLM_API_KEY` | your key; omit for endpoints that take no auth | — |
| `BIOGNOSIA_LLM_TIMEOUT` | seconds per API call | `300` |
| `BIOGNOSIA_LLM_ENABLE_COT` | `false` strips `<think>` blocks from answers | `true` |
| `BIOGNOSIA_LLM_TEMPERATURE` / `_TOP_P` | omitted entirely when unset, which is what reasoning models that reject them need | unset |
| `BIOGNOSIA_LLM_MAX_TOKENS` / `_MAX_COMPLETION_TOKENS` | output cap; setting the latter makes the former ignored | unset |

If `BIOGNOSIA_LLM_API_KEY` is unset, the provider falls back to the conventional
variables — `OPENAI_API_KEY` / `GROQ_API_KEY` / `ANTHROPIC_API_KEY`, then
`LLM_API_KEY`. Worth knowing if you already have one of those exported: it will
be used silently. When a `BIOGNOSIA_LLM_BASE_URL` is set and no key is found
anywhere, a placeholder is sent, since local endpoints typically require none.

**Server** — MCP mode only.

| Variable | Purpose | Default |
|---|---|---|
| `MCP_LLM_MODE` | `sampling` or `standalone` | `sampling` |
| `MCP_RAG_CONFIG_PATH` | config file path | `config/rag-web.conf` |
| `MCP_LOG_LEVEL` / `MCP_LOG_FILE` | logging | `INFO` / console |
| `MCP_TEST_MODE` | `1` skips all databases, models and GPU; handshake only | `0` |
| `MCP_TEST_PROMPT` | prompt used for the test-mode round-trip | a fixed "reply PONG" string |
| `MCP_AUTH_TOKEN` | require `Authorization: Bearer <token>` on `/mcp` | off |
| `MCP_PROTOCOL_VERSION_FALLBACK` | version to negotiate when the client omits or sends an unknown one | `2025-11-25` |
| `MCP_RESPONSE_LOG_DIR` | directory for full-payload dumps | off |
| `MCP_MAX_CONCURRENT_TOOL_CALLS` | cap on in-flight tool calls (0 = unlimited) | `8` |
| `MCP_MAX_BODY_BYTES` | reject larger `/mcp` bodies | `1048576` |

**GPU, models and paths**

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace access token. Used at **run time** by the Python path; at **build time** by Docker (`--build-arg HF_TOKEN=…`), where the finished image needs none. |
| `BIOGNOSIA_EMBEDDING_DEVICE` | sets all three embedding devices at once |
| `BIOGNOSIA_EMBEDDING_CHUNK_DEVICE` / `_LABEL_DEVICE` / `_STAGE2_DEVICE` | per-model override |
| `BIOGNOSIA_RERANK_DEVICE` / `BIOGNOSIA_RERANK_STAGE1_DEVICE` | reranker devices |
| `BIOGNOSIA_RERANK_NUM_WORKERS` / `_STAGE1_NUM_WORKERS` | reranker pool sizes — see below |
| `BIOGNOSIA_QUERY_LOG_DIR` | per-query debug logs (empty = off) |
| `BIOGNOSIA_NGRAM_STATISTICS_FILE` | n-gram entity statistics TSV |
| `BIOGNOSIA_HOST_PORT` | published host port (Docker only) |
| `BIOGNOSIA_GPU_DEVICE` | CDI device string (Docker only) — see below |

## Choosing a GPU and sizing the rerank pools

Each rerank worker is a **separate process** holding its own CUDA context and
model copy, so the two `*_NUM_WORKERS` variables drive memory directly. Model
residency, measured on an H200:

| Component | Per unit | 4 + 4 workers |
|-----------|----------|---------------|
| main process (embedding + stage-2 model) | ~3.7 GB | 3.7 GB |
| stage-2 rerank worker | ~1.1 GB | 4.3 GB |
| stage-1 rerank worker | ~0.5 GB | 2.1 GB |
| **resident total** | | **~10 GB** |

> ⚠️ **Do not size your GPU from that table — it counts only the weights sitting
> in memory, not the activations while reranking.** Measured on a real query
> against a full corpus with `4 + 4` workers, peak usage reached **about 84 GB**,
> returning to baseline afterwards. The peak comes from the reranker running
> large batches over a large candidate pool: stage 1 scores up to
> `candidate_top_k` (2000) passages at `stage1_batch_size` 512 ×
> `stage1_max_length` 512 across its workers, and stage 2 runs at `batch_size`
> 512 × `max_length` 1024. On a card with less memory than that, reduce
> `stage1_batch_size` and `batch_size` first (they dominate), then
> `candidate_top_k` and `stage1_top_k`, then the worker counts.
>
> Peak scales with how much the corpus actually returns, so a small or empty
> knowledge base will look far cheaper than a production one. Budget from the
> batch settings, not from a quiet run.

The worker defaults (8 + 8) hold about 16.5 GB resident. `rag-web.conf`'s own
guidance is "2-4 for GPU (due to VRAM limits)", so 4 + 4 is a reasonable
starting point on a shared card; end-to-end latency is dominated by the LLM
calls rather than by reranking, so smaller pools cost little on typical queries.

**Selecting a card.** By default every GPU is visible and the `*_DEVICE`
variables pick among them by ordinal — the normal approach, and the only one
that can spread models across cards:

```dotenv
BIOGNOSIA_EMBEDDING_DEVICE=cuda:2
BIOGNOSIA_RERANK_DEVICE=cuda:2
BIOGNOSIA_RERANK_STAGE1_DEVICE=cuda:2
BIOGNOSIA_RERANK_NUM_WORKERS=4
BIOGNOSIA_RERANK_STAGE1_NUM_WORKERS=4
```

Under Docker, `BIOGNOSIA_GPU_DEVICE` optionally restricts which cards the
container can see at all — useful on a machine shared with other people, since
then no misconfiguration can allocate on someone else's GPU. It **renumbers** the
devices: a container given one card sees it as `cuda:0` whatever its physical
index.

```dotenv
BIOGNOSIA_GPU_DEVICE=nvidia.com/gpu=2   # only this card is visible…
BIOGNOSIA_EMBEDDING_DEVICE=cuda:0       # …and inside, it is cuda:0
BIOGNOSIA_RERANK_DEVICE=cuda:0
BIOGNOSIA_RERANK_STAGE1_DEVICE=cuda:0
```

Pick one convention or the other. Mixing them (`…gpu=2` together with `cuda:2`)
fails with *invalid device ordinal*.

## Tests

Run from the repository root:

```bash
pip install -r testing-env/requirements-dev.txt
pytest
```

The suite is hermetic: it needs no GPU, no databases, no LLM and not even torch.
It covers the MCP protocol surface, the configuration and environment-override
logic, the LLM-mode switch, and — in most detail — citation resolution: bracket
normalisation, label-to-passage mapping, same-paper grouping, dropping labels the
model invented, and the guarantee that citations keep their position and identity
through the rewrite.

## Data availability

**This repository contains the retrieval and serving code only.**

- The biomedical **knowledge base** — the contents of Milvus, Neo4j,
  Elasticsearch and MongoDB — is **not distributed here**. It is hosted
  separately; configure its endpoints in the `FILL THESE IN` block of your
  `.env`. Those endpoints are not published yet, which is why every value in that
  block is a placeholder.
- The repository contains **no corpus-construction or ingestion code**. Nothing
  here builds, populates or updates a knowledge base; it only queries one.
- No language model weights are included. Generation runs against an endpoint you
  supply.

Retrieval quality depends entirely on the knowledge base behind those endpoints.
Pointed at an empty database, the system starts and answers, but with no
supporting passages.

## Licence

MIT — see [`LICENSE`](LICENSE).

Portions of the retrieval code are adapted from
[LightRAG](https://github.com/HKUDS/LightRAG) (MIT). The pre-trained models this
software downloads carry their own licences, including one that is
**non-commercial**. See [`NOTICE`](NOTICE) for the full attribution.

## How to cite

The accompanying paper is under submission; no DOI has been issued yet. Until
then, please cite the software using the metadata in
[`CITATION.cff`](CITATION.cff).
