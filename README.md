# Biognosia RAG

Retrieval-augmented generation over a biomedical literature knowledge base.

Ask a question in natural language; the system finds relevant passages across a
vector index, a knowledge graph and a keyword index, reranks them, and asks a
language model to answer using only those passages — returning the answer with a
citation for every claim, each resolving to the paper and passage it came from.

You do not need to build or host the knowledge base, nor to ask us for access to
it. It runs on our infrastructure; this repository ships a tunnel client that
reaches it from your machine, and the read-only credentials are published in
`.env.example`. What you provide is a GPU and an API key for the language model
— see [Configure](#4-configure) for the providers this is verified against.

---

## Contents

- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [How a query is answered](#how-a-query-is-answered)
- [Requirements](#requirements)
- [Configuration reference](#configuration-reference)
- [Running on macOS](#running-on-macos)
- [Choosing a GPU and sizing the rerank pools](#choosing-a-gpu-and-sizing-the-rerank-pools)
- [Tests](#tests)
- [Data availability](#data-availability)
- [Licence](#licence)
- [How to cite](#how-to-cite)

---

## Quickstart

Five steps, copy-pasteable.

### 1. Clone

```bash
git clone https://github.com/tinnlab/biognosia-rag.git
cd biognosia-rag
```

### 2. Open the tunnel to the knowledge base

```bash
docker compose -f docker-compose.tunnel.yml up -d
```

This opens five local listeners that reach our databases over
`wss://tunnel.tinnguyen-lab.com`:

| Service | Local address |
|---|---|
| Milvus | `127.0.0.1:18110` |
| Neo4j | `127.0.0.1:18111` |
| Redis | `127.0.0.1:18112` |
| MongoDB | `127.0.0.1:18113` |
| Elasticsearch | `127.0.0.1:18114` |

Leave it running while you query. Stop it with
`docker compose -f docker-compose.tunnel.yml down`.

The lab hosts these databases so that reproducing the paper does not require
rebuilding a very large corpus. They exist to support the paper rather than as
permanent infrastructure, so access may end at some point; the repository
carries the current details for as long as it is available, and we are glad to
hear from anyone who needs access beyond that.

> The tunnels are published on **`127.0.0.1` only**, deliberately — they are not
> exposed to your network. One consequence: if you run *your* application inside
> a container, it cannot reach them at `localhost`, because inside a container
> `localhost` is the container. Put your container on the tunnel's `biognosia`
> network and address the services by name (`milvus:18110`, `neo4j:18111`, …),
> or set `external: true` on that network in `docker-compose.tunnel.yml` and
> point it at your own. The comment at the bottom of that file spells it out.

### 3. Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate

# torch must come from the CUDA 12.4 wheel index, not PyPI, or you get a CPU build
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# NLTK corpora. Not a pip dependency — without them, entity matching quietly degrades
python -m nltk.downloader punkt punkt_tab stopwords \
    averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

### 4. Configure

```bash
cp .env.example .env
```

**One line to fill in**: an API key for the language model.

The knowledge-base settings are already done — **read-only and deliberately
public**, published with the paper so anyone can reproduce our results, so there
is no access to request and nothing there to edit.

Keep the key out of anything you commit — `.env` is gitignored for that reason.

#### 4a. OpenAI `gpt-5` (recommended)

This is the configuration the pipeline is verified end-to-end against, and it is
what `.env.example` already ships — paste your key into the last line and you
are done:

```dotenv
BIOGNOSIA_LLM_PROVIDER=openai
BIOGNOSIA_LLM_MODEL=gpt-5
BIOGNOSIA_LLM_BASE_URL=https://api.openai.com/v1
BIOGNOSIA_LLM_API_KEY=<your-openai-api-key>      # <- this one
```

Get the key from [platform.openai.com](https://platform.openai.com) under
**API keys**. It is shown once, so copy it there and then; OpenAI keys begin
with `sk-`.

#### 4b. Another provider — e.g. Groq

Any OpenAI-compatible endpoint works. Change all four `BIOGNOSIA_LLM_*` lines
**together** — `BASE_URL` is required rather than optional, because the provider
setting chooses the client, not the endpoint, so on its own it would send your
key to OpenAI.

Groq serves OpenAI's open-weight `gpt-oss-120b`:

```dotenv
BIOGNOSIA_LLM_PROVIDER=groq
BIOGNOSIA_LLM_MODEL=openai/gpt-oss-120b
BIOGNOSIA_LLM_BASE_URL=https://api.groq.com/openai/v1
BIOGNOSIA_LLM_API_KEY=<your-groq-api-key>      # <- this one
```

Get the key from [console.groq.com](https://console.groq.com) under **API
Keys**. Groq keys begin with `gsk_`. The model id must carry the `openai/`
prefix — that is Groq's identifier for it, and the bare name is not valid.

> ⚠️ **Groq's free tier is not enough to run this.** It caps
> `openai/gpt-oss-120b` at **8,000 tokens per minute**, and a single query needs
> far more than that in one request: measured against the live knowledge base,
> the routing call alone asked for 8,615 tokens and the final generation call
> for 26,541. Both come back as `HTTP 413 … rate_limit_exceeded`, which is a
> per-request size rejection rather than a throttle — waiting does not help, and
> the query fails outright. Use a paid tier, or any provider whose per-request
> budget comfortably exceeds ~30k tokens.

### 5. Ask a question

```bash
python examples/query.py "What do published studies report about the sensitivity of BRCA1-mutant breast cancer cells to PARP inhibitors?"
```

The first run downloads about 5 GB of model weights and loads them onto the GPU,
so give it a few minutes. Later runs reuse the cache.

You get the answer with a citation marker for each claim, followed by the
passages it was grounded in:

```
Breast cancer cells with homozygous BRCA1 mutations are reported to be extremely
sensitive to PARP inhibitors ... [C3][C9]

--- 20 supporting passage(s) ---
[C3] chunk-2b1ad3b781031d8cb820e444b9f51891e6a5f27a-000003
[C9] chunk-642f9229d63323879f018dabcea011e7b79fccd6-000002
...
```

### Using it from your own code

`examples/query.py` is deliberately small; the whole integration is:

```python
from src_rag_web.app import RAGApp

app = RAGApp(config_path="config/rag-web.conf", skip_llm=False)
try:
    await app.initialize()                   # loads models, connects to the stores
    result = await app.query(query_text=question, mode="auto")
finally:
    await app.close()                        # always: initialize() starts GPU workers

print(result["response"])                    # answer, citing passages as [C1], [C2], ...
for ref in result["references"]:
    print(ref["handle"], ref["id"])          # -> C1 chunk-<paper hash>-<n>
```

Two things worth copying rather than paraphrasing:

- **`close()` belongs in a `finally`.** `initialize()` starts reranker worker
  processes before it connects to the databases, and those processes outlive the
  parent. Skipping cleanup on the error path leaks several GB of VRAM.
- **`skip_llm=False`** is what makes the process build its own LLM provider from
  the `BIOGNOSIA_LLM_*` settings.

The first 40 hex characters of a passage id are the paper's identifier, so
`references` is enough to trace any claim back to its source.

## Architecture

Five stores back the retrieval pipeline, all reached through the tunnel:

| Store | Role | Required? |
|-------|------|-----------|
| Milvus | dense vector index over passages and entities | yes |
| Redis | passage text, and the entity → passage index | yes |
| Neo4j | biomedical knowledge graph (entities and their relations) | yes |
| Elasticsearch | BM25 / hybrid keyword search | soft — retrieval falls back to the vector and graph paths |
| MongoDB | paper metadata used to render citations | soft — answers still return, but citations degrade to raw passage text |

Two cross-encoder rerankers and three embedding models run locally on your GPU.
The language model does not: it is called over the API with your key.

This package performs **no writes** to any store, and the credentials you are
given are read-only.

## How a query is answered

1. **Routing.** The model is offered two retrieval tools and decides whether the
   question needs full retrieval, entity lookup only, or a direct answer. Asking
   what the *literature reports* reliably routes to full retrieval; asking what
   role something plays may not.
2. **Entity extraction.** Biomedical entities in the question are matched
   against the knowledge graph — first by n-gram matching over entity labels
   embedded with MedCPT, falling back to vector search over the entity index.
3. **Graph expansion.** Matched entities are expanded over `REGULATES`,
   `INTERACTS_WITH` and `PART_OF` edges in Neo4j, then reranked and cut to a
   score threshold so expansion does not dilute the context.
4. **Parallel retrieval.** Three workers run concurrently: the graph path, the
   keyword path (model-expanded keywords → BM25 → reciprocal rank fusion), and
   the vector path (dense search, optionally with query expansion, HyDE and
   sub-question decomposition).
5. **Merge.** Results are merged by passage id, tracking which retrieval sources
   found each one, and downselected in a rank-balanced way so no single source
   dominates.
6. **Two-stage reranking.** A small cross-encoder (`ms-marco-TinyBERT-L2-v2`)
   filters the candidate pool; a stronger one (`jina-reranker-v2`) scores the
   survivors. Both run as pools of separate processes, which is what makes the
   worker counts a VRAM decision.
7. **Context building.** Surviving passages are labelled `[C1]`, `[C2]`, … The
   model never sees raw passage identifiers.
8. **Generation and grounding.** The model answers citing those labels, and each
   label maps back to a passage id you can trace to its paper.

## Requirements

**Hardware**

- An NVIDIA GPU. The models occupy roughly **10 GB** with the recommended 4 + 4
  rerank workers, but **peak usage while reranking a real query reached about
  84 GB** in our measurement — see
  [sizing](#choosing-a-gpu-and-sizing-the-rerank-pools) before choosing a card,
  and turn the batch settings down if you have less.
- ~10 GB of disk for the model weights.

**Software**

- Python 3.11 and CUDA 12.4 drivers. Linux is the supported platform; see
  [macOS](#running-on-macos) if you are on a Mac.
- Docker, to run the tunnel client.

**Credentials**

- An API key for generation — the only one you supply, and the quickstart walks
  you through it. Any OpenAI-compatible provider works, but it needs a
  per-request budget of roughly 30k tokens, which rules out some free tiers.
  The knowledge-base accounts are read-only, public, and already in
  `.env.example`.
- A HuggingFace token — strongly recommended, not strictly required. None of the
  models below is gated, so anonymous download works, but the reranker starts
  many worker processes that each contact the Hub, and anonymous requests are
  rate-limited; an HTTP 429 during model load becomes a startup failure.

**Models downloaded on first run**

| Model | Role | Licence |
|---|---|---|
| [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | passage and query embeddings | MIT |
| [`ncbi/MedCPT-Query-Encoder`](https://huggingface.co/ncbi/MedCPT-Query-Encoder) | entity-label embeddings | public domain (NLM/NCBI) |
| [`jinaai/jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | second-stage reranking | **CC-BY-NC-4.0 — non-commercial only** |
| [`cross-encoder/ms-marco-TinyBERT-L2-v2`](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2) | first-stage reranking | Apache-2.0 |

> **Licence notice.** The second-stage reranker is licensed for research and
> evaluation only. Nothing here enforces that — the model downloads without any
> agreement step — so complying with it is your responsibility. For commercial
> use, substitute a differently-licensed reranker via `[rerank]` in
> `config/rag-web.conf`. All four licences are recorded in [`NOTICE`](NOTICE).

## Configuration reference

Defaults live in `config/rag-web.conf`, which documents every retrieval and
reranking tunable inline. Environment variables override them, and only
variables that are set and non-empty take effect.

The knowledge-base variables are not listed here: they ship pre-filled in
`.env.example` to match the tunnel, with read-only public credentials, and there
is nothing to change. `.env.example` documents each one inline if you need the
detail.

**LLM**

| Variable | Purpose | Default |
|---|---|---|
| `BIOGNOSIA_LLM_PROVIDER` | `openai` \| `groq` \| `anthropic` \| `ollama`. `openai` and `groq` share the same OpenAI-compatible client. | `openai` |
| `BIOGNOSIA_LLM_MODEL` | model id, exactly as the provider spells it | `gpt-5` |
| `BIOGNOSIA_LLM_API_KEY` | API key | — (yours; see the quickstart) |
| `BIOGNOSIA_LLM_BASE_URL` | where requests go; **must include `/v1`** | `https://api.openai.com/v1` |
| `BIOGNOSIA_LLM_TIMEOUT` | seconds per API call | `300` |
| `BIOGNOSIA_LLM_ENABLE_COT` | `false` strips `<think>` blocks from answers | `true` |
| `BIOGNOSIA_LLM_MAX_COMPLETION_TOKENS` | output cap | `8192` |
| `BIOGNOSIA_LLM_TEMPERATURE` / `_TOP_P` / `_MAX_TOKENS` | left unset — see below | unset |

Three details worth knowing before you change any of these:

- **The provider does not choose the endpoint.** `openai` and `groq` both select
  the same OpenAI-compatible client, and neither sets a URL of its own, so
  `BASE_URL` is what actually routes the request. Change the provider without it
  and your key goes to `api.openai.com`.
- **An unset parameter is dropped, not merely defaulted.** Leaving `TEMPERATURE`
  and `TOP_P` unset means the values parts of the pipeline try to pass
  (`temperature=0.0`) never reach the API either. That is deliberate: it keeps
  the defaults working with any model, including reasoning models that reject
  `temperature` outright.
- **`MAX_COMPLETION_TOKENS` converts rather than coexists.** Setting it turns any
  `max_tokens` the pipeline passes into `max_completion_tokens`, so the
  deprecated parameter is never sent. Leaving both unset would let it through.

If `BIOGNOSIA_LLM_API_KEY` is unset, the provider falls back to `GROQ_API_KEY` /
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, then `LLM_API_KEY` — worth knowing if you
already have one of those exported, as it will be used silently.

**GPU, models and paths**

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace access token, used at run time when models download |
| `BIOGNOSIA_EMBEDDING_DEVICE` | sets all three embedding devices at once |
| `BIOGNOSIA_EMBEDDING_CHUNK_DEVICE` / `_LABEL_DEVICE` / `_STAGE2_DEVICE` | per-model override |
| `BIOGNOSIA_RERANK_DEVICE` / `BIOGNOSIA_RERANK_STAGE1_DEVICE` | reranker devices |
| `BIOGNOSIA_RERANK_NUM_WORKERS` / `_STAGE1_NUM_WORKERS` | reranker pool sizes — see below |
| `BIOGNOSIA_QUERY_LOG_DIR` | per-query debug logs (empty = off) |
| `BIOGNOSIA_NGRAM_STATISTICS_FILE` | n-gram entity statistics TSV |

## Running on macOS

If you want to run Biognosia on your macOS machine, you can — but expect it to
be slow, and treat it as a way to explore the code rather than to do real work.
macOS has no CUDA, so the two cross-encoder rerankers, which are the expensive
half of the pipeline, fall back to the CPU. The quickstart above assumes Linux
and needs three changes.

**1. Install torch from PyPI, not the CUDA index.** The `cu124` wheel index in
step 3 has no macOS builds, so that command fails with "no matching
distribution". Replace it with:

```bash
pip install torch==2.5.1
```

The rest of step 3 — `requirements.txt` and the NLTK corpora — is unchanged.

**2. Put the embedding models on Apple Silicon, the rerankers on the CPU.** The
three embedding models support Metal via `mps`. The rerankers do not: they fall
back to the CPU whenever CUDA is absent, whatever you set. Say so explicitly, so
the configuration reflects what actually happens:

```dotenv
BIOGNOSIA_EMBEDDING_DEVICE=mps
BIOGNOSIA_RERANK_DEVICE=cpu
BIOGNOSIA_RERANK_STAGE1_DEVICE=cpu
```

**3. Shrink the rerank workload.** The defaults are sized for a data-centre GPU
and will exhaust a laptop. Each worker is a separate process holding its own
model copy, and on macOS that is *system RAM* rather than VRAM — the ~10 GB
resident and ~84 GB peak quoted in [sizing](#choosing-a-gpu-and-sizing-the-rerank-pools)
both come out of the same memory your machine runs on. Start small:

```dotenv
BIOGNOSIA_RERANK_NUM_WORKERS=1
BIOGNOSIA_RERANK_STAGE1_NUM_WORKERS=1
```

and in `config/rag-web.conf` cut the batch settings, which dominate the peak —
`stage1_batch_size` and `batch_size` from 512 down to 16-32 first, then
`candidate_top_k` from 2000 and `stage1_top_k` from 1024 to a few hundred.

Everything else works unchanged: the tunnel client runs under Docker Desktop,
and the language model is an API call rather than a local model, so generation
is unaffected.

> **Not verified.** The Linux/CUDA path is what we test and measure; the macOS
> path above follows from how the code selects devices, not from a run we have
> done. Expect reranking to take minutes rather than seconds, and please tell us
> what you hit.

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
> against the full corpus with `4 + 4` workers, peak usage reached **about
> 84 GB**, returning to baseline afterwards. The peak comes from the reranker
> running large batches over a large candidate pool: stage 1 scores up to
> `candidate_top_k` (2000) passages at `stage1_batch_size` 512 ×
> `stage1_max_length` 512, and stage 2 runs at `batch_size` 512 ×
> `max_length` 1024. On a smaller card, reduce `stage1_batch_size` and
> `batch_size` first (they dominate), then `candidate_top_k` and `stage1_top_k`,
> then the worker counts.

The worker defaults (8 + 8) hold about 16.5 GB resident. `rag-web.conf`'s own
guidance is "2-4 for GPU (due to VRAM limits)", so 4 + 4 is a reasonable
starting point; end-to-end latency is dominated by the LLM calls rather than by
reranking, so smaller pools cost little.

By default every GPU is visible and the `*_DEVICE` variables pick among them by
ordinal (`cuda:2` = physical GPU 2), which is also the only way to spread models
across cards.

## Tests

Run from the repository root:

```bash
pip install -r testing-env/requirements-dev.txt
python -m pytest
```

`python -m pytest` rather than `pytest`: the package is imported from the
repository root rather than installed, and only the `-m` form puts the current
directory on `sys.path`. Plain `pytest` fails collection with
`ModuleNotFoundError: No module named 'src_rag_web'`.

The suite is hermetic: no GPU, no databases, no LLM, not even torch. It covers
the configuration and environment-override logic, credential redaction, and — in
most detail — citation resolution.

## Data availability

**This repository contains the retrieval and serving code only.**

- The biomedical **knowledge base** is not distributed here. It is hosted by us
  and reached through the tunnel client, using the read-only credentials
  published in `.env.example` — open to anyone, no request needed.
- The repository contains **no corpus-construction or ingestion code**. Nothing
  here builds, populates or updates a knowledge base; it only queries one.
- No language model weights are included. Generation runs against the API with
  your own key.

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
