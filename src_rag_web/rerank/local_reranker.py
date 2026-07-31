"""
Local reranker using HuggingFace cross-encoder models.

Supports running rerank models locally on GPU/CPU.
Adapted from plans/lightrag-code/rerank/rerankers.py
"""

import asyncio
import logging

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class LocalReranker:
    """
    Local reranking model using cross-encoder architecture.

    Supports models from HuggingFace:
    - BAAI/bge-reranker-v2-m3 (multilingual, recommended)
    - BAAI/bge-reranker-large
    - BAAI/bge-reranker-base
    - cross-encoder/ms-marco-MiniLM-L-12-v2
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda:0",
        max_length: int = 512,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """
        Initialize local reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to run model on (cuda:0, cuda:1, cpu)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for inference
            normalize: Apply sigmoid to normalize scores to [0, 1] range (default: True)
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize

        # Warn if CUDA requested but not available
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")

        self.model = None
        self.tokenizer = None
        self._initialized = False

        logger.info(
            f"LocalReranker initialized: model={model_name}, "
            f"device={self.device}, max_length={max_length}, batch_size={batch_size}, "
            f"normalize={normalize}"
        )

    def initialize(self):
        """Load model and tokenizer (lazy loading)."""
        if self._initialized:
            return

        try:
            logger.info(f"Loading reranker model: {self.model_name}")

            # Load tokenizer (trust_remote_code needed for some models like jina-reranker)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            # Load model (trust_remote_code needed for some models like jina-reranker)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()

            self._initialized = True
            logger.info(f"Reranker model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load reranker model {self.model_name}: {e}")
            raise

    def compute_scores(self, query: str, documents: list[str]) -> list[float]:
        """
        Compute relevance scores for query-document pairs.

        Args:
            query: The search query
            documents: List of document texts

        Returns:
            List of relevance scores (one per document)
        """
        if not self._initialized:
            self.initialize()

        if not documents:
            return []

        scores = []

        # Process in batches
        for batch_idx, i in enumerate(range(0, len(documents), self.batch_size)):
            batch_docs = documents[i : i + self.batch_size]

            try:
                # Create query-document pairs
                pairs = [[query, doc] for doc in batch_docs]

                # Tokenize
                with torch.no_grad():
                    inputs = self.tokenizer(
                        pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                    )

                    # Check for invalid token IDs before moving to device
                    input_ids = inputs["input_ids"]
                    vocab_size = self.tokenizer.vocab_size
                    if vocab_size is not None:
                        max_token_id = input_ids.max().item()
                        if max_token_id >= vocab_size:
                            logger.error(
                                f"CRITICAL: Batch {batch_idx + 1} contains invalid token ID {max_token_id} "
                                f"(vocab_size={vocab_size}). This will cause CUDA assertion failure."
                            )
                            logger.error(f"Batch size: {len(batch_docs)} documents")
                            logger.error(f"Query length: {len(query)} chars")
                            logger.error(f"Document lengths: {[len(doc) for doc in batch_docs[:5]]} (showing first 5)")
                            raise RuntimeError(
                                f"Invalid token ID {max_token_id} exceeds vocab_size {vocab_size} "
                                f"in batch {batch_idx + 1}"
                            )

                    # Move to device
                    inputs = inputs.to(self.device)

                    # Get model scores
                    outputs = self.model(**inputs)
                    # Convert to float32 before numpy (BFloat16 not supported by numpy)
                    batch_scores = outputs.logits.squeeze(-1).float().cpu().numpy()

                    # Handle single item case
                    if isinstance(batch_scores, np.ndarray) and batch_scores.ndim == 0:
                        batch_scores = [float(batch_scores)]
                    else:
                        batch_scores = batch_scores.tolist()

                    # Apply sigmoid normalization if enabled
                    if self.normalize:
                        batch_scores = [1.0 / (1.0 + np.exp(-score)) for score in batch_scores]

                    scores.extend(batch_scores)

            except Exception as e:
                logger.error(
                    f"CRITICAL: Batch {batch_idx + 1}/{(len(documents) + self.batch_size - 1) // self.batch_size} "
                    f"failed during reranking: {e}"
                )
                logger.error(f"Batch range: documents[{i}:{i + len(batch_docs)}]")
                raise

        return scores

    async def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = all)

        Returns:
            List of (document_index, relevance_score) tuples sorted by relevance
        """
        if not documents:
            return []

        try:
            # Run GPU computation in thread pool to avoid blocking event loop
            # This allows multiple workers to run GPU work in parallel
            scores = await asyncio.to_thread(self.compute_scores, query, documents)

            # Create (index, score) pairs
            indexed_scores = list(enumerate(scores))

            # Sort by score (descending)
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Limit to top_k if specified
            if top_k is not None:
                indexed_scores = indexed_scores[:top_k]

            logger.debug(f"Reranked {len(documents)} documents, returning top {len(indexed_scores)}")

            return indexed_scores

        except Exception as e:
            logger.error(f"CRITICAL: Local reranking failed: {e}")
            logger.error("Stopping query pipeline due to reranking failure")
            raise

    def __del__(self):
        """Cleanup when object is destroyed."""
        # Let PyTorch handle cleanup automatically
        # Manual CUDA operations here can corrupt state between queries
        pass
