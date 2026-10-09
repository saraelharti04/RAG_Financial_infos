"""
Évaluation du RAG avec RAGAS (brique 8).

Concept :
- On ne se fie plus à une seule query manuelle : on mesure des scores
  (faithfulness, answer relevancy, context precision) sur un petit dataset.
- Un LLM "juge" (ici Ollama qwen2.5:3b) note si la réponse est ancrée
  dans le contexte, pertinente, etc.

Limites V1 (à connaître pour le CV) :
- Juge 3B = approximatif (mieux qu'aucune métrique, moins fiable qu'un gros juge).
- Embeddings de métriques = même BGE que le retrieval (déjà local, pas de pull Ollama).
- Stoppe Streamlit avant de lancer (lock Qdrant path=).

Usage (venv + Ollama ouverts) :
    pip install -r requirements.txt
    python -m src.evaluation.run_ragas
    python -m src.evaluation.run_ragas --max-questions 2   # smoke test rapide
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness
from sentence_transformers import SentenceTransformer

from src.embeddings.embed_chunks import DEFAULT_MODEL_NAME
from src.generation.rag_answer import (
    DEFAULT_OLLAMA_MODEL,
    generate_answer,
    retrieve_hits,
)
from src.indexing.index_qdrant import (
    COLLECTION_NAME,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_QDRANT_PATH,
)
from src.retrieval.hybrid_search import BM25Index, load_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "questions.json"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "data" / "eval" / "ragas_results.json"


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset vide ou invalide : {path}")
    return data


def build_samples(
    questions: list[dict],
    *,
    top_k: int,
    bm25: BM25Index,
    embedder: SentenceTransformer,
) -> list[SingleTurnSample]:
    """
    Pour chaque question du dataset :
    1) retrieval hybride
    2) génération Ollama
    3) empaquete en SingleTurnSample pour RAGAS
    """
    samples: list[SingleTurnSample] = []

    for i, item in enumerate(questions, start=1):
        qid = item.get("id", f"q{i}")
        question = item["question"]
        company = item.get("company")
        ground_truth = item["ground_truth"]

        print(f"\n[{i}/{len(questions)}] {qid}")
        print(f"  Q: {question}")

        hits = retrieve_hits(
            question,
            company=company,
            top_k=top_k,
            candidates=20,
            use_rerank=False,
            embeddings_dir=DEFAULT_EMBEDDINGS_DIR,
            qdrant_path=DEFAULT_QDRANT_PATH,
            collection=COLLECTION_NAME,
            bm25_index=bm25,
            embedder=embedder,
        )
        answer = generate_answer(
            question,
            hits,
            llm="ollama",
            model=DEFAULT_OLLAMA_MODEL,
        )
        contexts = [h.text for h in hits]

        print(f"  → {len(contexts)} contexts | answer[:120]={answer[:120]!r}…")

        # Noms de champs RAGAS récents : user_input / response / retrieved_contexts / reference
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )

    return samples


def run_ragas(
    samples: list[SingleTurnSample],
    *,
    judge_model: str,
) -> dict:
    """Configure le juge Ollama + embeddings BGE, lance evaluate()."""
    print(f"\nJuge RAGAS : Ollama/{judge_model}")
    print(f"Embeddings métriques : {DEFAULT_MODEL_NAME}")

    # temperature=0 → jugements un peu plus stables d'un run à l'autre
    judge_llm = LangchainLLMWrapper(
        ChatOllama(model=judge_model, temperature=0)
    )
    # Réutilise BGE déjà téléchargé (évite ollama pull nomic-embed-text).
    metric_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=DEFAULT_MODEL_NAME)
    )

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
        ],
        llm=judge_llm,
        embeddings=metric_embeddings,
    )
    # EvaluationResult n'est pas un dict : moyennes dans _repr_dict
    return {k: float(v) for k, v in result._repr_dict.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évalue le RAG avec RAGAS + Ollama.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limite le nombre de questions (smoke test)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help="Modèle Ollama juge (défaut : même qwen2.5:3b)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    if args.max_questions is not None:
        questions = questions[: args.max_questions]

    print(
        "⚠️  Ferme Streamlit avant si tu l'as encore ouvert "
        "(Qdrant local = un seul process)."
    )
    print(f"Dataset : {len(questions)} questions depuis {args.questions}")

    print("Chargement BM25 + embedder…")
    bm25 = BM25Index(load_corpus(DEFAULT_EMBEDDINGS_DIR))
    embedder = SentenceTransformer(DEFAULT_MODEL_NAME)

    samples = build_samples(
        questions,
        top_k=args.top_k,
        bm25=bm25,
        embedder=embedder,
    )

    scores = run_ragas(samples, judge_model=args.judge_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "judge_model": args.judge_model,
        "generator_model": DEFAULT_OLLAMA_MODEL,
        "n_questions": len(samples),
        "scores": scores,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== SCORES RAGAS ===")
    for name, value in scores.items():
        print(f"  {name}: {value}")
    print(f"\nRésultats écrits dans {args.output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
