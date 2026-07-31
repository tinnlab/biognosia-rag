# Biognosia MCP — production image (app-only, CUDA/GPU).
#
# Runs ONLY the Biognosia app and connects out to EXISTING external databases
# (MongoDB / Milvus / Neo4j / Redis / Elasticsearch) via env vars. No databases
# are bundled. Embedding + rerank run on the GPU, so run this image with a GPU
# via CDI (--device nvidia.com/gpu=all; see docker-compose.yml / run.sh).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/share/nltk_data \
    HF_HOME=/home/app/.cache/huggingface

# Python + minimal system deps (curl for the HEALTHCHECK).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Torch from the CUDA 12.4 wheel index (GPU build).
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 1b) FlashAttention 2 — prebuilt wheel matched to this image (python 3.11 /
#     torch 2.5 / cu12 / cxx11abiFALSE). The base is a CUDA *runtime* image with
#     no nvcc, so we install a prebuilt wheel rather than compiling from source
#     (a plain `pip install flash-attn` would try to build and fail). The
#     jina-reranker-v2 modeling code uses flash-attn when present.
#     If torch / python / cuda ever change, update this wheel URL to match
#     (https://github.com/Dao-AILab/flash-attention/releases).
RUN python -m pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 1c) Disable the TorchScript JIT. `import flash_attn.ops.activations` (pulled in
#     transitively by the jina-reranker-v2 modeling code) runs a module-level
#     `torch.jit.script(...)` that SEGFAULTS with this torch 2.5.1 / flash-attn
#     2.7.4 combo — which would crash every reranker worker at load. Disabling the
#     JIT turns that decorator into a no-op (eager fallback); flash-attn's CUDA
#     attention kernels are compiled separately and are unaffected.
ENV PYTORCH_JIT=0

# 2) The rest of the dependency set.
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# 3) NLTK corpora baked into the image (names differ across NLTK versions —
#    download both the legacy and the *_eng / *_tab variants).
#    Retry: the NLTK server occasionally returns a transient HTTP error, and the
#    downloader's interactive "Retry?" prompt aborts a non-interactive build.
RUN for i in 1 2 3 4 5; do \
        python -m nltk.downloader -d "$NLTK_DATA" \
            punkt punkt_tab stopwords \
            averaged_perceptron_tagger averaged_perceptron_tagger_eng </dev/null \
        && break; \
        echo "nltk download attempt $i failed; retrying in 5s"; sleep 5; \
        [ "$i" = 5 ] && exit 1; \
    done

# 4) Bake every HF model loaded at startup into the image so the first boot needs
#    NO network. Without a baked cache the reranker's from_pretrained(...,
#    trust_remote_code=True) makes a live HF API call (transformers 4.57
#    model_info), and the many reranker worker processes doing that at once are
#    easily rate-limited (HTTP 429) -> "startup failed" -> restart loop.
#    The two rerankers load via trust_remote_code, so snapshot_download the whole
#    repo to cache their custom modeling .py alongside the weights. HF_HOME (set
#    above) places everything under /home/app/.cache, chowned to `app` below.
#
#    bge-m3's main revision ships ONLY pytorch_model.bin (no safetensors), but the
#    app forces `use_safetensors=True`. Online, transformers papers over this by
#    fetching an auto-converted safetensors from a hub side-revision — a live HF
#    call that offline mode blocks. So we convert bin -> safetensors ourselves,
#    into bge-m3's main snapshot dir, so the offline load finds it locally.
#
#    ARG HF_TOKEN authenticates the BUILD's own downloads. None of the four
#    models is gated, so an anonymous build can work, but anonymous Hub requests
#    are rate-limited and a 429 mid-build is a hard failure — so pass your own
#    token: `docker build --build-arg HF_TOKEN=hf_... .`. It is consumed inline
#    on the RUN only, never ENV'd or persisted into a layer, and the finished
#    image needs no token at run time.
#
#    Licence note: jina-reranker-v2-base-multilingual is CC-BY-NC-4.0
#    (non-commercial). See NOTICE for all four model licences.
ARG HF_TOKEN
RUN HF_TOKEN=$HF_TOKEN python - <<'PY'
import os, shutil
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file, load_file
from transformers import AutoModel

for repo in [
    "jinaai/jina-reranker-v2-base-multilingual",  # rerank stage 2 (trust_remote_code)
    "cross-encoder/ms-marco-TinyBERT-L2-v2",       # rerank stage 1 (trust_remote_code)
    "BAAI/bge-m3",                                  # chunk + stage-2 embeddings
    "ncbi/MedCPT-Query-Encoder",                    # label / entity-name embeddings
]:
    print(f"Baking {repo} ...", flush=True)
    snapshot_download(repo)

# bge-m3: its main revision ships ONLY pytorch_model.bin, but the app loads it
# with use_safetensors=True (transformers 4.57 + torch 2.5 actually REFUSES to
# torch.load a .bin at all, so safetensors is the only usable format). Offline,
# if from_pretrained can't find model.safetensors in the *main* snapshot it
# enters transformers' safetensors auto-conversion, which calls the HF API
# (model_info) -> OfflineModeIsEnabled -> startup crash. So materialise
# model.safetensors from bge-m3's OWN weights, into the refs/main-resolved
# snapshot (not a guessed glob) under the exact name from_pretrained resolves.
# Model identity is preserved and asserted bit-equivalent, so the paper's model
# is never silently altered or swapped for a fallback.
model_dir = f"{os.environ['HF_HOME']}/hub/models--BAAI--bge-m3"
with open(f"{model_dir}/refs/main") as fh:
    main_rev = fh.read().strip()
snap = f"{model_dir}/snapshots/{main_rev}"
dst = os.path.join(snap, "model.safetensors")

if not os.path.exists(dst):
    print(f"Converting bge-m3 -> model.safetensors in main snapshot {main_rev} ...", flush=True)
    # Raw torch.load (weights_only) is permitted on torch 2.5; only transformers'
    # from_pretrained bin path is version-gated, so we go through safetensors.
    state = torch.load(os.path.join(snap, "pytorch_model.bin"),
                       map_location="cpu", weights_only=True)
    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, dst, metadata={"format": "pt"})

    # Integrity: the on-disk safetensors must equal bge-m3's bin weights exactly.
    disk = load_file(dst)
    assert disk.keys() == state.keys(), "bge-m3 key set changed during conversion"
    for k in state:
        assert torch.equal(disk[k], state[k]), f"bge-m3 weight mismatch at {k}"

    # Offline self-test: the app's exact runtime call (use_safetensors=True) must
    # now load with NO network. If this were going to hit the HF API it would
    # fail the build here instead of in production.
    AutoModel.from_pretrained(snap, use_safetensors=True, local_files_only=True)
    print(f"  wrote {dst} ({os.path.getsize(dst)} bytes); "
          f"verified bit-identical over {len(state)} tensors; offline load OK",
          flush=True)

# Drop any negative-cache markers so offline resolution can never short-circuit
# to the HF API for a file that is in fact present.
shutil.rmtree(f"{model_dir}/.no_exist", ignore_errors=True)
PY

# 5) Runtime offline: transformers / huggingface_hub must NEVER reach the HF API
#    at boot (this is what removes the model_info 429). Set AFTER the download
#    step above, which of course requires network.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 6) Application code + default config.
COPY src_rag_web/ ./src_rag_web/
COPY config/ ./config/

# Non-root runtime user. Pre-create the HuggingFace cache dir owned by `app` so
# the mounted named volume (docker-compose.yml) inherits app ownership and the
# non-root user can write the downloaded model weights into it.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/log "$HF_HOME" \
    && chown -R app:app /app /home/app/.cache
USER app

EXPOSE 8081

# Model weights are baked into the image (step 4), so boot no longer downloads
# them; the long start-period is kept as a safety margin for model load onto GPU.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=10 \
    CMD curl -fsS http://localhost:8081/health || exit 1

# Single worker is mandatory: sampling state lives in module-global memory.
CMD ["uvicorn", "src_rag_web.mcp_server:app", \
     "--host", "0.0.0.0", "--port", "8081", "--workers", "1"]
