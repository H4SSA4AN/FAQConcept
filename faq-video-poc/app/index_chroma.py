"""
Chroma database indexing and management.
"""

import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from loguru import logger
import uuid

from .settings import settings
from .embed import TextEmbedder


class ChromaIndexer:
    """Handles Chroma database operations for FAQ indexing."""

    def __init__(self, collection_name: str = "faq_collection"):
        """
        Initialize Chroma indexer.

        Args:
            collection_name: Name of the Chroma collection
        """
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedder = TextEmbedder()
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Chroma client and collection."""
        try:
            logger.info("Initializing Chroma client")

            # Configure Chroma settings
            chroma_settings = ChromaSettings(
                persist_directory=str(settings.chroma_persist_dir),
                is_persistent=True
            )

            self.client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir),
                settings=chroma_settings
            )

            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self._embedding_function
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self._embedding_function
                )
                logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise

    def _embedding_function(self, texts: List[str]) -> List[List[float]]:
        """Custom embedding function for Chroma."""
        embeddings = self.embedder.encode_batch(texts, normalize=True)
        return embeddings.tolist()

    def add_faqs(self, faqs_df: pd.DataFrame, batch_size: int = 100):
        """
        Add FAQ data to the Chroma collection.

        Args:
            faqs_df: DataFrame with columns: id, question, answer, category
            batch_size: Batch size for processing
        """
        try:
            logger.info(f"Adding {len(faqs_df)} FAQs to Chroma collection")

            # Prepare data
            documents = []
            metadatas = []
            ids = []

            for _, row in faqs_df.iterrows():
                # Combine question and answer for better search
                document = f"Question: {row['question']}\nAnswer: {row['answer']}"

                metadata = {
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row.get("category", "General"),
                    "id": str(row["id"])
                }

                documents.append(document)
                metadatas.append(metadata)
                ids.append(str(uuid.uuid4()))

            # Add to collection in batches
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))

                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )

                logger.debug(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")

            logger.info("Successfully added all FAQs to Chroma collection")

        except Exception as e:
            logger.error(f"Failed to add FAQs to Chroma: {e}")
            raise

    def search(self, query: str, n_results: int = 5, where: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search for similar FAQs in the collection.

        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filters

        Returns:
            Search results with documents, metadatas, and distances
        """
        try:
            logger.debug(f"Searching Chroma for: '{query}'")

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )

            return results

        except Exception as e:
            logger.error(f"Failed to search Chroma: {e}")
            raise

    def delete_collection(self):
        """Delete the current collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_dimension": self.embedder.dimension
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    def __repr__(self) -> str:
        return f"ChromaIndexer(collection='{self.collection_name}')"
