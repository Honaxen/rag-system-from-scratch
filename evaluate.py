"""
evaluate.py — Run RAG evaluation on rag-system-from-scratch outputs.

Usage:
    python3 evaluate.py --doc data/your_doc.txt --questions data/questions.json
    python3 evaluate.py --doc data/your_doc.txt --questions data/questions.json --out results/

How it works:
    1. Ingest and index the document (hybrid: FAISS + BM25)
    2. Ask each question via RAGGenerator
    3. Save { question, context, answer, ground_truth } to JSON
    4. Run rag-evaluation-framework metrics
    5. Print + save report
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, 'src')
from retrieval.vector_store import VectorStore
from generation.generator import RAGGenerator

# ── rag-evaluation-framework
RAG_EVAL_PATH = os.path.join(os.path.dirname(__file__), '..', 'rag-evaluation-framework')
sys.path.insert(0, RAG_EVAL_PATH)

try:
    from evaluator.pipeline import evaluate_dataset
    from evaluator.report import generate_reports
except ImportError:
    print("Error: rag-evaluation-framework not found.")
    print(f"Expected at: {RAG_EVAL_PATH}")
    print("Clone it: git clone https://github.com/Honaxen/rag-evaluation-framework")
    sys.exit(1)

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def _color(score: float) -> str:
    if score >= 0.8: return GREEN
    elif score >= 0.5: return YELLOW
    return RED


def _load_document(path: str) -> list[str]:
    """Load and chunk document into sentences."""
    import re
    with open(path, encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'\n+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for i in range(0, len(sentences), 4):
        chunk = ' '.join(sentences[i:i+4]).strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
    return chunks


def run_evaluation(doc_path: str, questions: list[dict], output_dir: str, model: str):
    print(f"\n{BOLD}RAG System Evaluation{RESET}")
    print(f"{DIM}Document : {doc_path}{RESET}")
    print(f"{DIM}Questions: {len(questions)}{RESET}\n")

    # Step 1: Index
    print(f"{CYAN}[1/3] Indexing document...{RESET}")
    chunks = _load_document(doc_path)
    vs = VectorStore()
    vs.build(chunks)
    generator = RAGGenerator(model=model)
    print(f"  → {vs.size} chunks indexed (hybrid: FAISS + BM25)\n")

    # Step 2: Ask questions
    print(f"{CYAN}[2/3] Running queries...{RESET}")
    dataset = []

    for i, q in enumerate(questions):
        question = q["question"]
        ground_truth = q.get("ground_truth", "")
        print(f"  [{i+1}/{len(questions)}] {question[:60]}...")

        results = vs.retrieve(question, top_k=3)
        context = [r["chunk"] for r in results]
        answer = generator.generate(question, context)

        dataset.append({
            "id": f"q{i+1}",
            "question": question,
            "context": context,
            "answer": answer,
            "ground_truth": ground_truth,
        })

    # Save raw output
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(output_dir, f"rag_output_{ts}.json")
    with open(raw_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\n  → Raw output: {raw_path}\n")

    # Step 3: Evaluate
    print(f"{CYAN}[3/3] Evaluating...{RESET}")
    eval_result = evaluate_dataset(dataset)
    agg = eval_result.aggregate

    print(f"\n{CYAN}{BOLD}{'─'*44}{RESET}")
    print(f"  {'Metric':<18} {'Score':>8}  {'Bar'}")
    print(f"{CYAN}{'─'*44}{RESET}")

    for name, score in [
        ("Faithfulness", agg.faithfulness),
        ("Relevance",    agg.relevance),
        ("Completeness", agg.completeness),
        ("Precision",    agg.precision),
    ]:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        c = _color(score)
        print(f"  {name:<18} {c}{score:>6.2f}{RESET}  {c}{bar}{RESET}")

    print(f"{CYAN}{'─'*44}{RESET}")
    c = _color(agg.overall)
    print(f"  {'Overall':<18} {c}{BOLD}{agg.overall:>6.2f}{RESET}")
    print(f"{CYAN}{BOLD}{'─'*44}{RESET}\n")

    json_path, html_path = generate_reports(eval_result, output_dir)
    print(f"{GREEN}Reports saved:{RESET}")
    print(f"  JSON → {json_path}")
    print(f"  HTML → {html_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--doc",       required=True, help="Path to .txt document")
    parser.add_argument("--questions", required=True, help="Path to questions JSON")
    parser.add_argument("--out",       default="eval_results", help="Output directory")
    parser.add_argument("--model",     default="gemma3:12b", help="Ollama model")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)

    run_evaluation(args.doc, questions, args.out, args.model)


if __name__ == "__main__":
    main()