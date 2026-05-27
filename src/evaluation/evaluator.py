"""
evaluator.py
------------
Evaluation metrics for RAG systems.

Classes:
    RAGEvaluator: Measure retrieval accuracy and hallucination.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEvaluator:
    """Evaluate retrieval quality and hallucination in RAG systems."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def grounding_score(self, answer: str, context: list) -> float:
        """
        Measure how grounded the answer is in the retrieved context.

        Score close to 1.0 = answer is grounded in context.
        Score close to 0.0 = answer may be hallucinated.

        Args:
            answer: Generated answer
            context: List of retrieved chunks used as context

        Returns:
            Cosine similarity score between answer and context
        """
        answer_embedding = self.model.encode([answer])
        context_embedding = self.model.encode([' '.join(context)])
        score = cosine_similarity(answer_embedding, context_embedding)[0][0]
        return float(score)

    def retrieval_accuracy(self, ground_truth: list, vector_store) -> dict:
        """
        Measure retrieval accuracy given ground truth labels.

        Args:
            ground_truth: List of {'query': str, 'correct_chunk': int}
            vector_store: VectorStore instance

        Returns:
            Dict with accuracy score and per-query results
        """
        correct = 0
        results = []

        for item in ground_truth:
            retrieved = vector_store.retrieve(item['query'], top_k=1)
            top_chunk = retrieved[0]['chunk']
            expected_chunk = vector_store.chunks[item['correct_chunk']]
            hit = top_chunk == expected_chunk
            if hit:
                correct += 1
            results.append({
                'query': item['query'],
                'hit': hit,
                'expected': expected_chunk[:80],
                'got': top_chunk[:80]
            })

        return {
            'accuracy': correct / len(ground_truth),
            'correct': correct,
            'total': len(ground_truth),
            'results': results
        }