#!/usr/bin/env python
"""Query the Biognosia knowledge base from plain Python, with your own LLM key.

This is the simplest way to use this repository: no MCP client, no Docker, no
server. It builds the RAG application directly, runs one query, and prints the
answer along with the passages it was grounded in.

Setup:

    cp .env.example .env      # then fill in the endpoints and your LLM key
    pip install -r requirements.txt
    python examples/query.py "What role does BRCA1 play in DNA repair?"

Configuration comes from the environment (see .env.example). If you use a `.env`
file, either export it into your shell or install python-dotenv and let this
script load it — it does so automatically when the package is available.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src_rag_web.app import RAGApp  # noqa: E402  (after sys.path setup)

DEFAULT_QUESTION = "What role does BRCA1 play in homologous recombination repair?"


def _load_dotenv() -> None:
    """Load `.env` if python-dotenv is installed; otherwise rely on the shell.

    Deliberately optional: python-dotenv is a convenience, not a dependency of
    the library itself, and the Docker path passes the environment in directly.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)


async def run(question: str) -> int:
    app = RAGApp(
        config_path=os.environ.get("MCP_RAG_CONFIG_PATH", str(REPO_ROOT / "config" / "rag-web.conf")),
        # skip_llm=False is the whole point: this process owns its LLM provider,
        # built from the [llm] config / BIOGNOSIA_LLM_* environment. (The MCP
        # server defaults to skip_llm=True and asks its client instead.)
        skip_llm=False,
    )

    # initialize() is INSIDE the try, deliberately. It starts the reranker worker
    # pools — separate processes, each holding a CUDA context — before it connects
    # to the stores, so a connection failure part-way through leaves those
    # processes running and holding GPU memory. They outlive this one (killing the
    # parent does not reap multiprocessing children), so without close() on the
    # error path a failed first run silently leaks several GB of VRAM. close()
    # guards every component individually and is safe on a partial init.
    try:
        # Loads the embedding and reranker models onto the GPU and connects to
        # every store, so the first call takes a while.
        await app.initialize()
        result = await app.query(query_text=question, mode="auto")
    finally:
        await app.close()

    print(result.get("response", ""))

    references = result.get("references") or []
    if references:
        print(f"\n--- {len(references)} supporting passage(s) ---")
        for ref in references:
            handle = ref.get("handle")
            # Passages the model was not shown have no handle; they are reported
            # for transparency but were not available for citation.
            marker = f"[{handle}]" if handle else "[not shown to the model]"
            print(f"{marker} {ref.get('id')}")
    else:
        print(
            "\n--- no supporting passages ---\n"
            "The knowledge base returned nothing for this query. That is expected "
            "against an empty database; otherwise, try a question about a specific "
            "gene, protein, pathway or disease."
        )
    return 0


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("MCP_LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _load_dotenv()

    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    try:
        return asyncio.run(run(question))
    except RuntimeError as exc:
        # Configuration problems (unfilled placeholders, no LLM configured) are
        # raised as RuntimeError with an actionable message. A traceback would
        # bury it.
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
