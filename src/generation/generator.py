"""
generator.py
------------
Answer generation using a local Ollama model.

Classes:
    RAGGenerator: Build prompts and generate answers from context.
"""

import requests


class RAGGenerator:
    """Generate answers from retrieved context using a local Ollama model."""

    def __init__(self, model: str = 'gemma3:12b',
                 ollama_url: str = 'http://127.0.0.1:11434'):
        self.model = model
        self.ollama_url = ollama_url

    def build_prompt(self, query: str, context: list) -> str:
        """Build a RAG prompt from query and context chunks."""
        context_text = '\n'.join(context)
        return f"""Answer the question based only on the context below.
Context:
{context_text}

Question: {query}
Answer:"""

    def generate(self, query: str, context: list) -> str:
        """
        Generate an answer from query and context.

        Args:
            query: User question
            context: List of retrieved text chunks

        Returns:
            Generated answer string
        """
        prompt = self.build_prompt(query, context)

        response = requests.post(
            f'{self.ollama_url}/api/chat',
            json={
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False
            },
            proxies={'http': None, 'https': None}
        )
        return response.json()['message']['content']