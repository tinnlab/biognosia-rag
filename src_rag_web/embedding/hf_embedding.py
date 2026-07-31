"""
HuggingFace embedding module for RAG query system.

Based on LightRAG llm/hf.py:144-182.

CRITICAL REQUIREMENTS (for LightRAG consistency):
1. Masked mean pooling (average over non-padding tokens only)
2. L2 normalization to unit length
3. Must match LightRAG's query-time embedding computation
"""

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class HuggingFaceEmbedding:
    """
    HuggingFace embedding model wrapper.

    Implements masked mean pooling + L2 normalization as required for LightRAG consistency.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """
        Initialize HuggingFace embedding model.

        Args:
            model_name: Model name (e.g., "BAAI/bge-m3", "ncbi/MedCPT-Query-Encoder")
            device: Device to use (cuda:0, cuda:1, cpu)
            max_length: Maximum sequence length
            batch_size: Batch size for embedding computation
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = None
        self.model = None
        self._initialized = False

    def _load_model(self):
        """Load model and tokenizer (lazy loading)."""
        if self._initialized:
            return

        try:
            logger.info(f"Loading embedding model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model with safetensors for security (torch 2.6+ vulnerability mitigation)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                use_safetensors=True,
            )

            # Detect device
            if self.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    self.device = "cpu"
            elif self.device == "mps":
                if not torch.backends.mps.is_available():
                    logger.warning("MPS not available, falling back to CPU")
                    self.device = "cpu"

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            self._initialized = True

            # Log model parameters
            logger.info(
                f"Embedding model loaded successfully - "
                f"Model: {self.model_name}, "
                f"Device: {self.device}, "
                f"Max length: {self.max_length}, "
                f"Batch size: {self.batch_size}"
            )

        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    def _masked_mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked mean pooling (average over non-padding tokens only).

        This is CRITICAL for consistency with LightRAG (LightRAG requirement).

        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        # Expand attention mask to match embedding dimensions
        # [batch_size, seq_len] -> [batch_size, seq_len, hidden_dim]
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings, weighted by attention mask
        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)

        # Sum attention mask (count of non-padding tokens)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

        # Average: divide by count
        embeddings = sum_embeddings / sum_mask

        return embeddings

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        L2 normalization to unit length.

        This is CRITICAL for consistency with LightRAG (LightRAG requirement).

        Args:
            embeddings: Embeddings [batch_size, hidden_dim]

        Returns:
            Normalized embeddings [batch_size, hidden_dim]
        """
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    async def embed(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        """
        Compute embeddings for texts.

        Implements the EXACT algorithm from LightRAG hf.py:144-182:
        1. Tokenize texts
        2. Forward pass through model
        3. Masked mean pooling (average over non-padding tokens)
        4. L2 normalization to unit length

        Args:
            texts: List of text strings to embed
            max_length: Override default max_length for this call (optional)

        Returns:
            Embeddings as numpy array [num_texts, embedding_dim]
        """
        if not texts:
            return np.array([])

        # Load model if not already loaded
        self._load_model()

        # Use provided max_length or default
        effective_max_length = max_length if max_length is not None else self.max_length

        try:
            # Tokenize texts
            encoded_texts = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=effective_max_length
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded_texts["input_ids"],
                    attention_mask=encoded_texts["attention_mask"],
                )

                # Get token embeddings (last hidden state)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded_texts["attention_mask"]

                # Step 1: Masked mean pooling
                embeddings = self._masked_mean_pooling(token_embeddings, attention_mask)

                # Step 2: L2 normalization
                embeddings = self._normalize_embeddings(embeddings)

            # Convert to numpy
            if embeddings.dtype == torch.bfloat16:
                embeddings = embeddings.to(torch.float32)

            embeddings_np = embeddings.cpu().numpy()

            logger.debug(f"Computed embeddings: {len(texts)} texts, shape={embeddings_np.shape}, device={self.device}")

            return embeddings_np

        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            raise

    def __call__(self, texts: list[str]) -> np.ndarray:
        """
        Synchronous wrapper for embed().

        Args:
            texts: List of text strings to embed

        Returns:
            Embeddings as numpy array
        """
        import asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run async function
        if loop.is_running():
            # If loop is already running, create a new task
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(self.embed(texts))
        else:
            return loop.run_until_complete(self.embed(texts))


class EmbeddingManager:
    """
    Manager for multiple embedding models.

    Handles both content embeddings (BGE-M3) and label embeddings (MedCPT).
    """

    def __init__(self, config: dict):
        """
        Initialize embedding manager from config.

        Args:
            config: Embedding configuration from rag.conf
        """
        self.config = config

        # Content embeddings (for chunks and entity descriptions)
        self.chunk_model_name = config.get("chunk_model", "BAAI/bge-m3")
        self.chunk_model_device = config.get("chunk_model_device", "cuda:0")
        self.chunk_model_max_length = int(config.get("chunk_model_max_length", 512))

        # Label embeddings (for entity names)
        self.label_model_name = config.get("label_model", "ncbi/MedCPT-Query-Encoder")
        self.label_model_device = config.get("label_model_device", "cuda:0")
        self.label_model_max_length = int(config.get("label_model_max_length", 256))

        # Stage 2 embeddings (for semantic community discovery)
        self.stage2_model_name = config.get("stage2_model", "BAAI/bge-m3")
        self.stage2_model_device = config.get("stage2_model_device", "cuda:0")
        self.stage2_model_max_length = int(config.get("stage2_model_max_length", 2048))

        # Batch size
        self.batch_size = int(config.get("batch_size", 32))

        # Log device configuration for debugging
        logger.info("EmbeddingManager initialized with devices:")
        logger.info(f"  Chunk model: {self.chunk_model_name} -> {self.chunk_model_device}")
        logger.info(f"  Label model: {self.label_model_name} -> {self.label_model_device}")
        logger.info(f"  Stage2 model: {self.stage2_model_name} -> {self.stage2_model_device}")

        # Lazy load models
        self._chunk_model = None
        self._label_model = None
        self._stage2_model = None

    @property
    def chunk_model(self) -> HuggingFaceEmbedding:
        """Get or create content embedding model."""
        if self._chunk_model is None:
            self._chunk_model = HuggingFaceEmbedding(
                model_name=self.chunk_model_name,
                device=self.chunk_model_device,
                max_length=self.chunk_model_max_length,
                batch_size=self.batch_size,
            )
        return self._chunk_model

    @property
    def label_model(self) -> HuggingFaceEmbedding:
        """Get or create label embedding model."""
        if self._label_model is None:
            self._label_model = HuggingFaceEmbedding(
                model_name=self.label_model_name,
                device=self.label_model_device,
                max_length=self.label_model_max_length,
                batch_size=self.batch_size,
            )
        return self._label_model

    @property
    def stage2_model(self) -> HuggingFaceEmbedding:
        """Get or create Stage 2 semantic community discovery embedding model."""
        if self._stage2_model is None:
            # If stage2_model uses the same model as chunk_model, reuse it to save GPU memory
            if self.stage2_model_name == self.chunk_model_name and self.stage2_model_device == self.chunk_model_device:
                logger.info(f"Stage 2 model reuses chunk model ({self.chunk_model_name}) to save GPU memory")
                # Reuse chunk model - max_length will be overridden in embed() call
                self._stage2_model = self.chunk_model
                logger.info(
                    f"Stage 2 uses max_length={self.stage2_model_max_length} "
                    f"(chunk model default: {self.chunk_model_max_length})"
                )
            else:
                # Load separate model if different
                logger.info(f"Stage 2 loading separate model: {self.stage2_model_name}")
                self._stage2_model = HuggingFaceEmbedding(
                    model_name=self.stage2_model_name,
                    device=self.stage2_model_device,
                    max_length=self.stage2_model_max_length,
                    batch_size=self.batch_size,
                )
        return self._stage2_model

    async def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """
        Embed text chunks using content model (BGE-M3).

        Args:
            texts: List of chunk texts

        Returns:
            Embeddings array
        """
        return await self.chunk_model.embed(texts)

    async def embed_labels(self, labels: list[str]) -> np.ndarray:
        """
        Embed entity labels using label model (MedCPT).

        Args:
            labels: List of entity names

        Returns:
            Embeddings array
        """
        return await self.label_model.embed(labels)

    async def embed_stage2(self, texts: list[str]) -> np.ndarray:
        """
        Embed Stage 2 shuffled queries using Stage 2 model (BGE-M3 with long context).

        Args:
            texts: List of shuffled text combinations

        Returns:
            Embeddings array
        """
        # Pass stage2 max_length to override the model's default
        return await self.stage2_model.embed(texts, max_length=self.stage2_model_max_length)

    async def embed_for_outlier_detection(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts for outlier detection using CLS token pooling.

        Uses Stage 2 model (bge-m3) but with CLS token pooling instead of
        masked mean pooling. CLS pooling better captures semantic differences
        for short texts, making outlier detection more effective.

        Args:
            texts: List of entity IDs or snippet texts

        Returns:
            Embeddings array using CLS token pooling
        """
        if not texts:
            return np.array([])

        # Use stage2 model (bge-m3)
        model = self.stage2_model
        model._load_model()

        try:
            encoded_texts = model.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.stage2_model_max_length
            ).to(model.device)

            with torch.no_grad():
                outputs = model.model(
                    input_ids=encoded_texts["input_ids"],
                    attention_mask=encoded_texts["attention_mask"],
                )

                # Use CLS token (first token) instead of masked mean pooling
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

                # L2 normalization
                cls_embeddings = model._normalize_embeddings(cls_embeddings)

            return cls_embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"Error during outlier detection embedding: {e}")
            raise

    async def embed_query(self, query: str, use_label_model: bool = False) -> np.ndarray:
        """
        Embed a single query.

        Args:
            query: Query text
            use_label_model: If True, use label model; otherwise use chunk model

        Returns:
            Query embedding (1D array)
        """
        if use_label_model:
            embeddings = await self.embed_labels([query])
        else:
            embeddings = await self.embed_chunks([query])

        return embeddings[0]

    def preload_models(self):
        """
        Eagerly load all embedding models into memory.

        This should be called during app initialization to avoid
        lazy loading delays during query execution.
        """
        logger.info("Preloading embedding models...")

        # Force model initialization by accessing properties and loading models
        logger.info(f"Loading chunk model: {self.chunk_model_name}")
        self.chunk_model._load_model()

        logger.info(f"Loading label model: {self.label_model_name}")
        self.label_model._load_model()

        # Access stage2_model property (triggers reuse logic or separate load)
        # This will either reuse chunk_model or load separately
        _ = self.stage2_model  # Trigger property access to set up reuse

        logger.info("All embedding models preloaded successfully")


def create_embedding_manager(config: dict) -> EmbeddingManager:
    """
    Create embedding manager from configuration.

    Args:
        config: Embedding configuration from rag.conf

    Returns:
        Embedding manager instance
    """
    return EmbeddingManager(config)
