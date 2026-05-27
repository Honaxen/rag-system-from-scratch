"""
chunker.py
----------
Document ingestion and chunking strategies.

Classes:
    DocumentChunker: Load, clean, and chunk text documents.
"""

import re


class DocumentChunker:
    """Load, clean, and chunk text documents."""

    def clean(self, text: str) -> str:
        """
        Clean raw document text.
        - Normalize newlines
        - Remove extra whitespace
        """
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def chunk_by_size(self, text: str, chunk_size: int = 200, overlap: int = 50) -> list:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: Input text
            chunk_size: Max characters per chunk
            overlap: Characters shared between consecutive chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += chunk_size - overlap
        return chunks

    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 3) -> list:
        """
        Split text into chunks by sentence boundaries.

        Args:
            text: Input text
            sentences_per_chunk: Number of sentences per chunk

        Returns:
            List of text chunks
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def load_and_chunk(self, text: str, strategy: str = 'sentences',
                       sentences_per_chunk: int = 3) -> list:
        """
        Full ingestion pipeline: clean + chunk.

        Args:
            text: Raw document text
            strategy: 'sentences' or 'size'
            sentences_per_chunk: Used when strategy='sentences'

        Returns:
            List of cleaned text chunks
        """
        cleaned = self.clean(text)
        if strategy == 'sentences':
            return self.chunk_by_sentences(cleaned, sentences_per_chunk)
        return self.chunk_by_size(cleaned)
