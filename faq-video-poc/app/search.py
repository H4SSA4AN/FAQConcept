"""
Main FAQ search functionality combining multiple vector databases.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from loguru import logger
import re

from .settings import settings
from .embed import TextEmbedder
from .index_chroma import ChromaIndexer


@dataclass
class SearchResult:
    """Represents a single search result."""
    question: str
    answer: str
    category: str
    score: float
    source: str  # 'chroma'
    metadata: Optional[Dict[str, Any]] = None


class FAQSearch:
    """Main FAQ search engine supporting multiple vector databases."""

    def __init__(self, use_chroma: bool = True):
        """
        Initialize the FAQ search engine.

        Args:
            use_chroma: Whether to use Chroma database
        """
        self.use_chroma = use_chroma
        self.embedder = TextEmbedder()

        # Initialize indexer
        self.chroma_indexer = None

        if self.use_chroma:
            try:
                self.chroma_indexer = ChromaIndexer()
                logger.info("Chroma indexer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Chroma indexer: {e}")
                self.use_chroma = False

        if not self.use_chroma:
            raise RuntimeError("Chroma database must be available")

    def _extract_primary_clause(self, query: str) -> str:
        """Return a trimmed query capturing the primary clause/intent."""
        if not query:
            return query
        text = query.strip()
        # Prefer first sentence up to . ? ! if present
        sentence = re.split(r"[.?!]", text, maxsplit=1)[0].strip()
        candidate = sentence if sentence else text
        # If too long, cut at first conjunction after ~12 tokens
        tokens = candidate.split()
        if len(tokens) > 14:
            for i in range(12, len(tokens)):
                if tokens[i].lower() in {"and", "or", "which", "that", "who", "when", "where", "why", "how"}:
                    candidate = " ".join(tokens[:i])
                    break
        return candidate

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
            # Initial retrieval size (raise top_k)
            limit = max(settings.app.max_results, 10)

        if threshold is None:
            threshold = settings.app.similarity_threshold

        logger.info(f"Searching for: '{query}' (limit={limit}, threshold={threshold})")

        all_results = []

        # Search Chroma with full and primary clause queries
        if self.use_chroma and self.chroma_indexer:
            try:
                initial_k = max(30, settings.app.max_results * 5)
                # Pull extra candidates for better reranking
                full_results = self._search_chroma(query, initial_k)
                all_results.extend(full_results)

                primary_query = self._extract_primary_clause(query)
                primary_results = []
                if primary_query and primary_query != query:
                    primary_results = self._search_chroma(primary_query, initial_k)
                    all_results.extend(primary_results)
            except Exception as e:
                logger.error(f"Chroma search failed: {e}")

        # Merge and rerank combining full vs primary scores per FAQ id
        by_id: Dict[str, Dict[str, Any]] = {}

        def add_results(results: List[SearchResult], weight: float):
            for r in results:
                faq_id = (r.metadata or {}).get('id') or r.question
                entry_type = (r.metadata or {}).get('entry_type', 'qa')
                # Base score with small question_only boost
                base = r.score + (0.10 if entry_type == 'question_only' else 0.0)
                if faq_id not in by_id:
                    by_id[faq_id] = {
                        'full': 0.0,
                        'primary': 0.0,
                        'best_meta': r,
                        'best_base': base,
                    }
                # Track highest base score source-wise
                if weight > 0.5:  # primary channel
                    by_id[faq_id]['primary'] = max(by_id[faq_id]['primary'], base)
                else:
                    by_id[faq_id]['full'] = max(by_id[faq_id]['full'], base)
                # Keep representative metadata from the highest base
                if base > by_id[faq_id]['best_base']:
                    by_id[faq_id]['best_base'] = base
                    by_id[faq_id]['best_meta'] = r

        # Split back out full and primary lists from all_results
        primary_query = self._extract_primary_clause(query)
        initial_k = max(30, settings.app.max_results * 5)
        full_results = self._search_chroma(query, initial_k) if (self.use_chroma and self.chroma_indexer) else []
        primary_results = self._search_chroma(primary_query, initial_k) if (self.use_chroma and self.chroma_indexer and primary_query and primary_query != query) else []

        add_results(full_results, weight=0.4)
        add_results(primary_results, weight=0.6)

        combined: List[SearchResult] = []
        for _id, rec in by_id.items():
            combined_score = 0.4 * rec['full'] + 0.6 * rec['primary']
            if combined_score >= threshold:
                base_result = rec['best_meta']
                combined.append(SearchResult(
                    question=base_result.question,
                    answer=base_result.answer,
                    category=base_result.category,
                    score=combined_score,
                    source=base_result.source,
                    metadata=base_result.metadata
                ))

        # Sort by combined score and trim to app max
        combined.sort(key=lambda x: x.score, reverse=True)
        final_results = combined[:settings.app.max_results]

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
            faqs_df = pd.read_csv(csv_path, encoding='utf-8')

            # Validate required columns
            required_columns = ['id', 'question', 'answer']
            missing_columns = [col for col in required_columns if col not in faqs_df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Add to indexer
            if self.use_chroma and self.chroma_indexer:
                self.chroma_indexer.add_faqs(faqs_df)
                logger.info("FAQs added to Chroma")

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

        return stats

    def __repr__(self) -> str:
        return f"FAQSearch(chroma={self.use_chroma})"
