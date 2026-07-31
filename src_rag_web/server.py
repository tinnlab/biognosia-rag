"""FastAPI server for the RAG query system with per-request LLM."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .app import RAGApp
from .llm import ExternalAPIProvider

logger = logging.getLogger(__name__)

# Global RAG app instance (initialized once at startup)
rag_app: RAGApp | None = None


def _setup_logging():
    """Configure logging to file and console."""
    log_file = os.environ.get("RAG_LOG_FILE", "")
    log_level = os.environ.get("RAG_LOG_LEVEL", "INFO")

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (if log file specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG infrastructure on startup, cleanup on shutdown."""
    global rag_app
    _setup_logging()
    config_path = os.environ.get("RAG_CONFIG_PATH", "config/rag-web.conf")
    logger.info(f"Starting RAG server with config: {config_path}")

    rag_app = RAGApp(config_path=config_path, skip_llm=True)
    await rag_app.initialize()
    logger.info("RAG server ready")

    yield

    logger.info("Shutting down RAG server...")
    if rag_app:
        await rag_app.close()


app = FastAPI(
    title="Biognosia RAG API",
    description="RAG query API with per-request LLM endpoint",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""

    query: str = Field(..., description="The query text")
    LLMAPIEndpoint: str = Field(..., description="URL of the LLM API endpoint")
    LLMAPIToken: str = Field(..., description="JWT/Bearer token for the LLM API")
    mode: str | None = Field(None, description="Query mode: naive, local, global, hybrid, mix, bypass, auto")
    top_k: int | None = Field(None, description="Number of entities/relationships to retrieve")
    enable_rerank: bool | None = Field(None, description="Enable reranking")
    response_type: str | None = Field(None, description="Response format type")


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""

    response: str
    metadata: dict | None = None


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG system with a per-request LLM endpoint.

    The caller provides their own LLM API endpoint and auth token.
    The RAG infrastructure (DBs, embeddings, rerankers) is shared across requests.
    """
    if rag_app is None or not rag_app._initialized:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    provider = ExternalAPIProvider(
        endpoint=request.LLMAPIEndpoint,
        token=request.LLMAPIToken,
    )

    try:
        # Build kwargs from optional parameters
        kwargs = {}
        if request.top_k is not None:
            kwargs["top_k"] = request.top_k
        if request.enable_rerank is not None:
            kwargs["enable_rerank"] = request.enable_rerank
        if request.response_type is not None:
            kwargs["response_type"] = request.response_type

        result = await rag_app.query(
            query_text=request.query,
            mode=request.mode,
            llm_provider=provider,
            **kwargs,
        )

        # Exclude bulky fields from metadata - references already contain chunk content
        exclude_keys = {"response", "chunks", "entities", "relationships"}
        return QueryResponse(
            response=result.get("response", ""),
            metadata={k: v for k, v in result.items() if k not in exclude_keys},
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        await provider.close()


@app.get("/health")
async def health():
    """Health check endpoint."""
    if rag_app is None:
        return {"status": "starting"}
    if not rag_app._initialized:
        return {"status": "initializing"}
    return {"status": "healthy"}
