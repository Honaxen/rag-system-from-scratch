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
Hybrid Index    — FAISS (dense) + BM25 (sparse)
   |        |
dense      sparse
   |        |
   └───┬────┘
       |
  RRF Fusion
       |
       v
Retrieval       — find relevant chunks for a query
   |
   v
Generation      — use chunks as context to generate answer
   |
   v
Evaluation      — measure faithfulness, relevance, completeness, precision
```

---

## Project Structure

```
rag-system-from-scratch/
├── notebooks/
│   ├── 01_ingestion.ipynb
│   ├── 02_retrieval.ipynb
│   ├── 03_generation.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_real_document.ipynb
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   │   └── vector_store.py  — Hybrid: FAISS + BM25 + RRF
│   ├── generation/
│   └── evaluation/
├── data/
│   ├── raw/
│   ├── processed/
│   └── questions.json
├── evaluate.py              — RAG evaluation script
└── README.md
```

---

## Getting Started

```bash
pip install -r requirements.txt
ollama serve
ollama pull gemma3:12b
```

### Run notebooks
```bash
jupyter notebook
```

### Run Evaluation
```bash
python3 evaluate.py --doc data/raw/sample_document.txt --questions data/questions.json
```

Output:
```
────────────────────────────────────────────
  Metric             Score  Bar
────────────────────────────────────────────
  Faithfulness        1.00  ████████████████████
  Relevance           0.11  ██░░░░░░░░░░░░░░░░░░
  Completeness        1.00  ████████████████████
  Precision           0.20  ████░░░░░░░░░░░░░░░░
────────────────────────────────────────────
  Overall             0.58
```

---

## Stack

Python · FAISS · BM25 · sentence-transformers · Ollama · pytest

---

## What I Learned

RAG is not one thing — it is a pipeline with four distinct components,
and each one affects the final answer quality.

Hybrid search beats pure semantic search.
FAISS alone misses exact keyword matches. BM25 alone misses semantic relationships.
Reciprocal Rank Fusion combines both without needing to normalize scores.

Evaluation reveals what intuition hides.
High faithfulness (1.00) means no hallucination.
Low precision (0.20) means retrieved chunks contain noise.
You can't fix what you can't see.

Chunking strategy directly impacts retrieval.
A chunk that cuts mid-sentence loses context and produces noisy embeddings.

---

## Results

| Metric | Score |
|--------|-------|
| Faithfulness | 1.00 |
| Relevance | 0.11 |
| Completeness | 1.00 |
| Precision | 0.20 |
| Overall | 0.58 |

---

## Related Projects

- [rag-evaluation-framework](https://github.com/Honaxen/rag-evaluation-framework) — the evaluation tool used here
- [document-agent](https://github.com/Honaxen/document-agent) — production-ready RAG agent

---

## Author

[Honaxen](https://github.com/Honaxen)