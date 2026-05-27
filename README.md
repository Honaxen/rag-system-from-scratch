# RAG System From Scratch

A Retrieval-Augmented Generation system built from the ground up.
No LangChain abstractions — every component is implemented and explained.

---

## What is RAG?

RAG combines two things:
- **Retrieval**: find relevant documents from a knowledge base
- **Generation**: use those documents as context to answer questions

Without RAG, LLMs hallucinate.
With RAG, answers are grounded in real documents.

---

## Pipeline

```
Document
   |
   v
Ingestion       — load, clean, chunk
   |
   v
Embeddings      — convert chunks to vectors
   |
   v
Vector Store    — index and store embeddings (FAISS)
   |
   v
Retrieval       — find relevant chunks for a query
   |
   v
Generation      — use chunks as context to generate answer
   |
   v
Evaluation      — measure hallucination and retrieval quality
```

---

## Project Structure

```
rag-system-from-scratch/
├── notebooks/
│   ├── 01_ingestion.ipynb
│   ├── 02_retrieval.ipynb
│   ├── 03_generation.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
├── data/
│   ├── raw/
│   └── processed/
└── README.md
```

---

## Stack

Python · FAISS · sentence-transformers · transformers · PyPDF2

---

## What I Learned

RAG is not one thing — it is a pipeline with four distinct components,
and each one affects the final answer quality.

Chunking strategy directly impacts retrieval.
A chunk that cuts mid-sentence loses context and produces noisy embeddings.

Retrieval quality is measurable.
100% accuracy on this dataset showed that semantic search with FAISS
finds the right chunk even without exact keyword matching.

Grounding score revealed that the model answered from context, not memory.
This is the core value of RAG — it turns a hallucinating model
into a document-grounded question answering system.

The hardest part was not the code — it was making all components work together:
local embeddings, vector store, local LLM, and evaluation in one pipeline.

---

## Results

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | 100% |
| Hallucination Detected | None |
| Grounding Score | 0.66 - 0.81 |

---

## Author

[Honaxen](https://github.com/Honaxen)