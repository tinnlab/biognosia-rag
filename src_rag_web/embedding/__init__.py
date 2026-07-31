"""Embedding module for RAG query system."""

from .hf_embedding import (
    EmbeddingManager,
    HuggingFaceEmbedding,
    create_embedding_manager,
)

__all__ = [
    "HuggingFaceEmbedding",
    "EmbeddingManager",
    "create_embedding_manager",
]
