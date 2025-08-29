"""
Text embedding functionality using sentence transformers.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from loguru import logger

from .settings import settings


class TextEmbedder:
    """Handles text embedding using sentence transformers."""

    def __init__(self, model_name: str = None):
        """
        Initialize the embedder with a sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name or settings.embedding.model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of text strings
            normalize: Whether to normalize the embeddings

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            logger.debug(f"Encoding {len(texts)} text(s)")
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text into an embedding.

        Args:
            text: Text string to encode
            normalize: Whether to normalize the embedding

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embeddings = self.encode([text], normalize=normalize)
        return embeddings[0]

    def encode_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encode texts in batches for better performance.

        Args:
            texts: List of text strings to encode
            batch_size: Size of each batch
            normalize: Whether to normalize the embeddings

        Returns:
            Numpy array of embeddings
        """
        try:
            logger.debug(f"Encoding {len(texts)} texts in batches of {batch_size}")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=True
            )
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to encode texts in batches: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def __repr__(self) -> str:
        return f"TextEmbedder(model='{self.model_name}', dimension={self.dimension})"
