"""
Chroma database indexing and management.
"""

import pandas as pd
import chromadb
from typing import List, Dict, Any, Optional
from loguru import logger
import uuid
import numpy as np

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
            logger.debug(f"Chroma persist directory: {settings.chroma_persist_dir}")

            # Create Chroma client
            self.client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir)
            )
            logger.debug("Chroma client created successfully")

            # Get or create collection
            try:
                logger.debug(f"Attempting to get collection: {self.collection_name}")
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception as e:
                # Collection doesn't exist or other error, create new one
                logger.info(f"Collection '{self.collection_name}' not found: {e}")
                logger.info(f"Creating new collection: {self.collection_name}")
                self.collection = self.client.create_collection(name=self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"ChromaDB version: {getattr(chromadb, '__version__', 'unknown')}")
            raise

    def add_faqs(self, faqs_df: pd.DataFrame, batch_size: int = 100):
        """
        Add FAQ data to the Chroma collection.

        Args:
            faqs_df: DataFrame with columns: id, question, answer, category
            batch_size: Batch size for processing
        """
        try:
            logger.info(f"Adding {len(faqs_df)} FAQs to Chroma collection")

            # Ensure collection exists and is properly initialized
            if self.collection is None:
                logger.info("Collection not initialized, initializing...")
                self._initialize_client()

            if self.collection is None:
                raise RuntimeError("Failed to initialize collection")

            # Prepare data
            documents = []
            embeddings = []
            metadatas = []
            ids = []

            for _, row in faqs_df.iterrows():
                # Create two documents: question-only (higher weight) and question+answer
                question_only_doc = f"Question: {row['question']}"
                question_answer_doc = f"Question: {row['question']}\nAnswer: {row['answer']}"

                base_metadata = {
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row.get("category", "General"),
                    "id": str(row["id"]),
                    "answer__url": row.get("answer__url", "")
                }

                # Question-only entry (type=question_only) for stronger question intent matching
                documents.append(question_only_doc)
                qo_meta = dict(base_metadata)
                qo_meta["entry_type"] = "question_only"
                metadatas.append(qo_meta)
                ids.append(str(uuid.uuid4()))

                # Question+Answer entry (type=qa)
                documents.append(question_answer_doc)
                qa_meta = dict(base_metadata)
                qa_meta["entry_type"] = "qa"
                metadatas.append(qa_meta)
                ids.append(str(uuid.uuid4()))

            # Compute embeddings; apply simple weighting by repeating question-only entries
            logger.debug("Computing embeddings for all documents (with question prioritization)")
            document_embeddings = self.embedder.encode_batch(documents, normalize=True)
            embeddings = document_embeddings.tolist()

            # Add to collection in batches
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))

                self.collection.add(
                    documents=documents[i:end_idx],
                    embeddings=embeddings[i:end_idx],
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

            # Ensure collection exists
            if self.collection is None:
                raise RuntimeError("Collection not initialized")

            # Compute embedding for the query
            query_embedding = self.embedder.encode_single(query, normalize=True).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
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
            # Reset the collection object to None so it gets reinitialized
            self.collection = None
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            # Ensure collection exists and is properly initialized
            if self.collection is None:
                logger.info("Collection not initialized, initializing...")
                self._initialize_client()

            if self.collection is None:
                raise RuntimeError("Failed to initialize collection")

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
