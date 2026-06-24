"""
vector_store.py
---------------
Hybrid retrieval: FAISS (dense) + BM25 (sparse) + RRF fusion.

Upgrade from v1:
  v1 — FAISS-only semantic search
  v2 — BM25 + FAISS combined with Reciprocal Rank Fusion (RRF)
"""

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ─────────────────────────────────────────────
# BM25 Retriever
# ─────────────────────────────────────────────

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def build(self, chunks: list[str]) -> None:
        self.chunks = chunks
        tokenized = [self._tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


# ─────────────────────────────────────────────
# FAISS Retriever
# ─────────────────────────────────────────────

class FAISSRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks: list[str]) -> None:
        self.chunks = chunks
        embeddings = self.model.encode(chunks).astype(np.float32)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_emb = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_emb, top_k)
        return [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0]))]


# ─────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def _rrf(bm25_results, faiss_results, k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(faiss_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# VectorStore — unified interface (backwards compatible)
# ─────────────────────────────────────────────

class VectorStore:
    """
    Hybrid retrieval: FAISS (dense) + BM25 (sparse) + RRF fusion.

    Backwards compatible with v1 — existing code using
    VectorStore.build() and VectorStore.retrieve() works unchanged.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.faiss = FAISSRetriever(model_name)
        self.bm25 = BM25Retriever()
        self.chunks = []

    def build(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.faiss.build(chunks)
        self.bm25.build(chunks)

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        """
        Hybrid retrieval — returns list of {"chunk": str, "distance": float}.
        Backwards compatible with v1 output format.
        """
        if not self.chunks:
            raise ValueError("Index not built. Call build() first.")

        candidate_k = min(top_k * 3, len(self.chunks))
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
        faiss_results = self.faiss.retrieve(query, top_k=candidate_k)
        fused = _rrf(bm25_results, faiss_results)

        return [
            {"chunk": self.chunks[idx], "distance": round(1.0 - score, 6)}
            for idx, score in fused[:top_k]
            if idx < len(self.chunks)
        ]

    @property
    def size(self) -> int:
        return len(self.chunks)
