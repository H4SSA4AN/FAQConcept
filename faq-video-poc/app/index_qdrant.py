"""
Qdrant database indexing and management.
"""

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
from loguru import logger
import uuid

from .settings import settings
from .embed import TextEmbedder


class QdrantIndexer:
    """Handles Qdrant database operations for FAQ indexing."""

    def __init__(self, collection_name: str = "faq_collection"):
        """
        Initialize Qdrant indexer.

        Args:
            collection_name: Name of the Qdrant collection
        """
        self.collection_name = collection_name
        self.client = None
        self.embedder = TextEmbedder()
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client."""
        try:
            logger.info("Initializing Qdrant client")

            # Initialize Qdrant client
            self.client = QdrantClient(
                host=settings.database.qdrant_host,
                port=settings.database.qdrant_port
            )

            # Create collection if it doesn't exist
            self._ensure_collection_exists()

            logger.info("Qdrant client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if necessary."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")

                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedder.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    def add_faqs(self, faqs_df: pd.DataFrame, batch_size: int = 100):
        """
        Add FAQ data to the Qdrant collection.

        Args:
            faqs_df: DataFrame with columns: id, question, answer, category
            batch_size: Batch size for processing
        """
        try:
            logger.info(f"Adding {len(faqs_df)} FAQs to Qdrant collection")

            points = []

            for _, row in faqs_df.iterrows():
                # Create embedding for the question
                question_embedding = self.embedder.encode_single(row["question"])

                # Prepare payload (metadata)
                payload = {
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row.get("category", "General"),
                    "id": str(row["id"])
                }

                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=question_embedding.tolist(),
                    payload=payload
                )

                points.append(point)

            # Add points in batches
            for i in range(0, len(points), batch_size):
                end_idx = min(i + batch_size, len(points))
                batch = points[i:end_idx]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

                logger.debug(f"Added batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")

            logger.info("Successfully added all FAQs to Qdrant collection")

        except Exception as e:
            logger.error(f"Failed to add FAQs to Qdrant: {e}")
            raise

    def search(self, query: str, limit: int = 5, filter_conditions: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search for similar FAQs in the collection.

        Args:
            query: Search query
            limit: Number of results to return
            filter_conditions: Filter conditions for search

        Returns:
            Search results with points and scores
        """
        try:
            logger.debug(f"Searching Qdrant for: '{query}'")

            # Create query embedding
            query_embedding = self.embedder.encode_single(query)

            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=filter_conditions
            )

            return search_results

        except Exception as e:
            logger.error(f"Failed to search Qdrant: {e}")
            raise

    def delete_collection(self):
        """Delete the current collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    def __repr__(self) -> str:
        return f"QdrantIndexer(collection='{self.collection_name}')"
