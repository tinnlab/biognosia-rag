"""FastAPI server exposing the RAG system over the Model Context Protocol.

Run with:
    uvicorn src_rag_web.mcp_server:app --host 0.0.0.0 --port 8081 --workers 1

The server keeps per-session sampling state in module-global memory, so it MUST
run with a single worker.

Environment variables:
    MCP_LLM_MODE         "sampling" (default) — generate nothing locally; ask the
                         connected MCP client for every completion via
                         sampling/createMessage.
                         "standalone" — call an LLM endpoint directly, using the
                         [llm] config / BIOGNOSIA_LLM_* environment. No
                         sampling-capable client needed.
    MCP_RAG_CONFIG_PATH  Path to the RAG config file (default: config/rag-web.conf)
    MCP_LOG_FILE         Optional path to a log file
    MCP_LOG_LEVEL        Log level (default: INFO)
    MCP_TEST_MODE        "1" to skip RAGApp/DB/GPU init and serve a minimal
                         protocol round-trip only, for checking the MCP
                         handshake without databases, models or a GPU.

See README.md for the full list (knowledge-base endpoints, LLM settings, auth).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# NOTE: `from .app import RAGApp` is deliberately deferred into the non-test
# branch of `lifespan` below. Importing RAGApp pulls in torch/transformers and
# the whole retrieval stack; keeping it out of the module top level lets the
# test image (MCP_TEST_MODE=1) run on a slim CPU base with no torch installed.
from .mcp_route import router as mcp_router

logger = logging.getLogger(__name__)


def _setup_logging():
    log_file = os.environ.get("MCP_LOG_FILE", "")
    log_level = os.environ.get("MCP_LOG_LEVEL", "INFO")

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


def _test_mode() -> bool:
    return os.environ.get("MCP_TEST_MODE", "").strip() in ("1", "true", "yes", "on")


_LLM_MODES = ("sampling", "standalone")


def _llm_mode() -> str:
    """How the server obtains generation.

    sampling   — (default) the server has no LLM of its own and asks the
                 connected MCP client for every completion via
                 sampling/createMessage. RAGApp is built with skip_llm=True and
                 a per-request provider is supplied for each tool call.
    standalone — the server calls an LLM endpoint itself, using the [llm]
                 config / BIOGNOSIA_LLM_* environment. No sampling-capable
                 client is required; a plain HTTP client can drive it.
    """
    mode = os.environ.get("MCP_LLM_MODE", "").strip().lower() or "sampling"
    if mode not in _LLM_MODES:
        raise RuntimeError(
            f"Invalid MCP_LLM_MODE={mode!r}; expected one of: {', '.join(_LLM_MODES)}"
        )
    return mode


# Validation itself lives in config.py so that BOTH entry points get it: the
# plain-Python path (examples/query.py) reaches it through RAGApp.initialize().
# Re-exported here because this module is where the standalone mode is chosen.
def _validate_llm_config(llm_cfg: dict) -> None:
    from .config import validate_llm_config

    validate_llm_config(llm_cfg)


def _require_llm_sdk(provider: str) -> None:
    from .config import require_llm_sdk

    require_llm_sdk(provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG infrastructure once at startup, store on app.state."""
    _setup_logging()

    if _test_mode():
        # Skip the whole RAGApp/DB/GPU initialization. Used to validate the client
        # sampling handshake locally without the external databases or a GPU.
        logger.warning(
            "MCP_TEST_MODE is enabled — skipping RAGApp initialization. "
            "query_rag will perform a minimal sampling round-trip only."
        )
        app.state.rag_app = None
        yield
        return

    # Deferred import: keeps torch/transformers out of the test image.
    from .app import RAGApp

    llm_mode = _llm_mode()
    config_path = os.environ.get("MCP_RAG_CONFIG_PATH", "config/rag-web.conf")
    logger.info(f"Starting MCP server with config: {config_path} (LLM mode: {llm_mode})")

    if llm_mode == "standalone":
        # Validate before initialize(): see _validate_llm_config. load_config is
        # just configparser plus environment lookups, cheap enough to run twice.
        from .config import load_config

        llm_cfg = load_config(config_path).get("llm", {})
        _validate_llm_config(llm_cfg)
        _require_llm_sdk((llm_cfg.get("provider") or "").strip().lower())
        logger.info(
            "Standalone LLM: provider=%s model=%s base_url=%s timeout=%ss (api_key %s)",
            llm_cfg.get("provider"),
            llm_cfg.get("model"),
            llm_cfg.get("base_url") or "<client default>",
            llm_cfg.get("timeout"),
            "set" if llm_cfg.get("api_key") else "unset",
        )

    rag_app = RAGApp(config_path=config_path, skip_llm=(llm_mode == "sampling"))
    await rag_app.initialize()
    app.state.rag_app = rag_app
    logger.info("MCP server ready")

    yield

    logger.info("Shutting down MCP server...")
    await rag_app.close()


app = FastAPI(
    title="Biognosia RAG — MCP Server",
    description="Biomedical retrieval-augmented generation served over the Model Context Protocol",
    lifespan=lifespan,
)

app.include_router(mcp_router)


@app.get("/health")
async def health():
    """Health check endpoint.

    Returns 503 until the app is actually ready so that Docker/orchestrator
    readiness gates (and the HEALTHCHECK start-period) behave correctly while the
    GPU models and database connections come up.
    """
    rag_app = getattr(app.state, "rag_app", None)
    if _test_mode():
        # In test mode there is no RAGApp; the server is ready once it boots.
        return {"status": "healthy", "mode": "test"}
    if rag_app is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    if not rag_app._initialized:
        return JSONResponse(status_code=503, content={"status": "initializing"})
    return {"status": "healthy"}
