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

TBD — will be updated after all components are complete.

---

## Author

[Honaxen](https://github.com/Honaxen)