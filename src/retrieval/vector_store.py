"""
vector_store.py
---------------
FAISS-based vector store for semantic retrieval.

Classes:
    VectorStore: Embed, index, and retrieve text chunks.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    """FAISS-based vector store for semantic retrieval."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks: list) -> None:
        """
        Embed chunks and build FAISS index.

        Args:
            chunks: List of text chunks
        """
        self.chunks = chunks
        embeddings = self.model.encode(chunks).astype(np.float32)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 2) -> list:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (chunk, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        query_embedding = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)

        return [
            {'chunk': self.chunks[i], 'distance': float(d)}
            for i, d in zip(indices[0], distances[0])
        ]