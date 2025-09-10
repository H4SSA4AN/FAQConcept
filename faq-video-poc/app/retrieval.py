"""
Lightweight retrieval with bi-encoder recall + cross-encoder rerank.

Developer notes:
- Keeps multi-qa-MiniLM-L6-cos-v1 as the bi-encoder (unchanged)
- Cross-encoder only reorders candidates and provides a confidence signal
- With ~30 docs, we can rerank all docs cheaply on CPU
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import re

import numpy as np
from sentence_transformers import CrossEncoder
from loguru import logger

from .settings import settings
from .index_chroma import ChromaIndexer

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False


# Cached singletons
_chroma: ChromaIndexer = None
_cross: CrossEncoder = None


def _first_sent(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    parts = re.split(r"[.?!]", text.strip())
    s = parts[0].strip() if parts else text.strip()
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _get_chroma() -> ChromaIndexer:
    global _chroma
    if _chroma is None:
        _chroma = ChromaIndexer()
    return _chroma


def _get_cross() -> CrossEncoder:
    global _cross
    if _cross is None:
        logger.info("Loading cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross


def _lexical_score(query: str, doc_text: str) -> float:
    if not HAS_RAPIDFUZZ:
        return 0.0
    try:
        # Lowercase only for lexical, keep punctuation
        return fuzz.token_set_ratio(query.lower(), doc_text.lower()) / 100.0
    except Exception:
        return 0.0


def answer(query: str, k: int = 10) -> Dict[str, Any]:
    """
    Returns:
      {"mode":"answer","question":..., "answer":..., "score":float}
      OR
      {"mode":"suggest","candidates":[{"question":..., "answer":..., "score":float}, ...]}
    """
    chroma = _get_chroma()

    # Recall via bi-encoder (Chroma) – use normalize_embeddings=True in indexer
    # Rerank pool: with ~30 docs, just fetch them all for simplicity
    results = chroma.search(query, n_results=50)
    docs = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    if not docs:
        return {"mode": "suggest", "candidates": []}

    # Convert cos distance to cosine similarity
    sims = [1.0 - d for d in distances]

    prelim_scores: List[float] = []
    rerank_pool: List[Dict[str, Any]] = []
    for meta, s in zip(docs, sims):
        q = meta.get("question", "")
        a = meta.get("answer", "")
        rid = meta.get("id")
        text_for_lex = f"{q}\n{_first_sent(a)}"
        lex = _lexical_score(query, text_for_lex)
        prelim = 0.8 * s + (0.2 * lex if HAS_RAPIDFUZZ else 0.0)
        prelim_scores.append(prelim)
        rerank_pool.append({"question": q, "answer": a, "prelim": prelim, "id": rid})

    # Take top-k prelim
    k = max(1, min(k, len(rerank_pool)))
    rerank_pool = [r for _, r in sorted(zip(prelim_scores, rerank_pool), key=lambda x: x[0], reverse=True)][:k]

    # Cross-encoder rerank
    cross = _get_cross()
    pairs = [(query, f"{r['question']}\n{_first_sent(r['answer'], 300)}") for r in rerank_pool]
    ce_scores = cross.predict(pairs).tolist()

    ranked = [
        {"question": r[1]["question"], "answer": r[1]["answer"], "score": float(r[0]), "id": r[1].get("id")}
        for r in sorted(zip(ce_scores, rerank_pool), key=lambda x: x[0], reverse=True)
    ]

    # Confidence gates
    CONF = 0.25
    MARGIN = 0.05
    best = ranked[0]["score"]
    second = ranked[1]["score"] if len(ranked) > 1 else -1.0

    if best < CONF or (second >= 0 and (best - second) < MARGIN):
        return {"mode": "suggest", "candidates": ranked[:3]}

    top = ranked[0]
    return {"mode": "answer", "question": top["question"], "answer": top["answer"], "score": top["score"], "id": top.get("id")}


