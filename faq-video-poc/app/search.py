"""
Main FAQ search functionality combining multiple vector databases.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from loguru import logger

from .settings import settings
from .embed import TextEmbedder
from .index_chroma import ChromaIndexer
from .index_qdrant import QdrantIndexer


@dataclass
class SearchResult:
    """Represents a single search result."""
    question: str
    answer: str
    category: str
    score: float
    source: str  # 'chroma' or 'qdrant'
    metadata: Optional[Dict[str, Any]] = None


class FAQSearch:
    """Main FAQ search engine supporting multiple vector databases."""

    def __init__(self, use_chroma: bool = True, use_qdrant: bool = True):
        """
        Initialize the FAQ search engine.

        Args:
            use_chroma: Whether to use Chroma database
            use_qdrant: Whether to use Qdrant database
        """
        self.use_chroma = use_chroma
        self.use_qdrant = use_qdrant
        self.embedder = TextEmbedder()

        # Initialize indexers
        self.chroma_indexer = None
        self.qdrant_indexer = None

        if self.use_chroma:
            try:
                self.chroma_indexer = ChromaIndexer()
                logger.info("Chroma indexer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Chroma indexer: {e}")
                self.use_chroma = False

        if self.use_qdrant:
            try:
                self.qdrant_indexer = QdrantIndexer()
                logger.info("Qdrant indexer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant indexer: {e}")
                self.use_qdrant = False

        if not self.use_chroma and not self.use_qdrant:
            raise RuntimeError("At least one vector database must be available")

    def search(self, query: str, limit: int = None, threshold: float = None) -> List[SearchResult]:
        """
        Search for FAQs matching the query.

        Args:
            query: Search query
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        if limit is None:
            limit = settings.app.max_results

        if threshold is None:
            threshold = settings.app.similarity_threshold

        logger.info(f"Searching for: '{query}' (limit={limit}, threshold={threshold})")

        all_results = []

        # Search Chroma
        if self.use_chroma and self.chroma_indexer:
            try:
                chroma_results = self._search_chroma(query, limit)
                all_results.extend(chroma_results)
            except Exception as e:
                logger.error(f"Chroma search failed: {e}")

        # Search Qdrant
        if self.use_qdrant and self.qdrant_indexer:
            try:
                qdrant_results = self._search_qdrant(query, limit)
                all_results.extend(qdrant_results)
            except Exception as e:
                logger.error(f"Qdrant search failed: {e}")

        # Filter by threshold and sort by score
        filtered_results = [r for r in all_results if r.score >= threshold]
        filtered_results.sort(key=lambda x: x.score, reverse=True)

        # Return top results
        final_results = filtered_results[:limit]

        logger.info(f"Found {len(final_results)} results above threshold {threshold}")
        return final_results

    def _search_chroma(self, query: str, limit: int) -> List[SearchResult]:
        """Search using Chroma indexer."""
        results = self.chroma_indexer.search(query, n_results=limit)

        search_results = []

        if results and results.get('documents'):
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Convert distance to similarity score (Chroma returns cosine distance)
                score = 1 - distance

                result = SearchResult(
                    question=metadata['question'],
                    answer=metadata['answer'],
                    category=metadata.get('category', 'General'),
                    score=score,
                    source='chroma',
                    metadata=metadata
                )
                search_results.append(result)

        return search_results

    def _search_qdrant(self, query: str, limit: int) -> List[SearchResult]:
        """Search using Qdrant indexer."""
        results = self.qdrant_indexer.search(query, limit=limit)

        search_results = []

        for point in results:
            result = SearchResult(
                question=point.payload['question'],
                answer=point.payload['answer'],
                category=point.payload.get('category', 'General'),
                score=point.score,
                source='qdrant',
                metadata=point.payload
            )
            search_results.append(result)

        return search_results

    def add_faqs_from_csv(self, csv_path: Optional[str] = None):
        """
        Load and index FAQs from CSV file.

        Args:
            csv_path: Path to CSV file (uses default if not provided)
        """
        if csv_path is None:
            csv_path = str(settings.faq_data_path)

        try:
            logger.info(f"Loading FAQs from: {csv_path}")
            faqs_df = pd.read_csv(csv_path)

            # Validate required columns
            required_columns = ['id', 'question', 'answer']
            missing_columns = [col for col in required_columns if col not in faqs_df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Add to indexers
            if self.use_chroma and self.chroma_indexer:
                self.chroma_indexer.add_faqs(faqs_df)
                logger.info("FAQs added to Chroma")

            if self.use_qdrant and self.qdrant_indexer:
                self.qdrant_indexer.add_faqs(faqs_df)
                logger.info("FAQs added to Qdrant")

            logger.info(f"Successfully indexed {len(faqs_df)} FAQs")

        except Exception as e:
            logger.error(f"Failed to load FAQs from CSV: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed data."""
        stats = {}

        if self.use_chroma and self.chroma_indexer:
            try:
                stats['chroma'] = self.chroma_indexer.get_collection_info()
            except Exception as e:
                logger.warning(f"Failed to get Chroma stats: {e}")

        if self.use_qdrant and self.qdrant_indexer:
            try:
                stats['qdrant'] = self.qdrant_indexer.get_collection_info()
            except Exception as e:
                logger.warning(f"Failed to get Qdrant stats: {e}")

        return stats

    def __repr__(self) -> str:
        return f"FAQSearch(chroma={self.use_chroma}, qdrant={self.use_qdrant})"
