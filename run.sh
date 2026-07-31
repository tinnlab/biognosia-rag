#!/usr/bin/env bash
# Convenience wrapper around `docker run` for the Biognosia RAG MCP server.
#
# Usage:
#   ./run.sh prod       Build + run the production CUDA image (needs .env + a GPU).
#   ./run.sh test       Build + run the slim CPU image with MCP_TEST_MODE=1, which
#                       skips all databases, models and GPU and answers only the
#                       MCP handshake — for checking protocol wiring.
#
# The build downloads four HuggingFace models. Set HF_TOKEN first — anonymous
# Hub requests are rate-limited and a 429 mid-build fails the build:
#
#   export HF_TOKEN=hf_...
#
# Uses `sudo docker` (the invoking user is not assumed to be in the docker group).
set -euo pipefail

MODE="${1:-prod}"
IMAGE_PROD="biognosia-rag:latest"
IMAGE_TEST="biognosia-rag:test"
NAME="biognosia-rag"

case "$MODE" in
  prod)
    if [ -z "${HF_TOKEN:-}" ]; then
      echo "warning: HF_TOKEN is not set. The build downloads four models from the" >&2
      echo "HuggingFace Hub; anonymous requests are rate-limited and a 429 will fail" >&2
      echo "the build. See the Requirements section of README.md." >&2
    fi
    # GPU via CDI (needs `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
    # on the host). --add-host lets the container reach services published on the
    # host via host.docker.internal.
    #
    # Deliberately NO volume over /home/app/.cache/huggingface: the models are
    # baked into the image and the container runs with HF_HUB_OFFLINE=1, so a
    # named volume would shadow the baked cache and crash startup.
    sudo docker build -f Dockerfile --build-arg HF_TOKEN="$HF_TOKEN" -t "$IMAGE_PROD" .
    sudo docker rm -f "$NAME" 2>/dev/null || true
    sudo docker run -d --name "$NAME" \
      --device nvidia.com/gpu=all \
      --add-host host.docker.internal:host-gateway \
      --env-file .env \
      -p 8081:8081 \
      --shm-size=1g \
      --security-opt no-new-privileges \
      --cap-drop ALL \
      --restart unless-stopped \
      "$IMAGE_PROD"
    echo "Started $NAME (prod). Health: curl http://localhost:8081/health"
    ;;
  test)
    sudo docker build -f Dockerfile.test -t "$IMAGE_TEST" .
    sudo docker rm -f "$NAME" 2>/dev/null || true
    sudo docker run -d --name "$NAME" \
      -e MCP_TEST_MODE=1 \
      -p 8081:8081 \
      "$IMAGE_TEST"
    echo "Started $NAME (test, no databases/GPU/LLM)."
    echo "Health: curl http://localhost:8081/health"
    echo "MCP endpoint: http://localhost:8081/mcp"
    ;;
  *)
    echo "Unknown mode: $MODE (expected 'prod' or 'test')" >&2
    exit 1
    ;;
esac
