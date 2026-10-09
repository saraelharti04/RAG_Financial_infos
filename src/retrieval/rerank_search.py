"""
Reranking cross-encoder des candidats hybrides (brique 6).

Concept :
- Dense + BM25 + RRF donnent un BON rappel (les bons chunks sont "quelque part"
  dans le top-20) mais un ranking encore bruité (ex. IT failures en #3).
- Un cross-encoder lit la PAIRE (question, chunk) ensemble et produit un score
  de pertinence plus fin → on reclasse le top-20 puis on garde le top-k.

Pourquoi cross-encoder et pas re-embedder :
- Bi-encoder (BGE) encode query et doc séparément → rapide, moins précis.
- Cross-encoder les lit ensemble → plus lent, bien meilleur pour le top du ranking.
  On ne l'applique donc QUE sur ~20 candidats, pas sur tout le corpus.

Modèle choisi : cross-encoder/ms-marco-MiniLM-L-6-v2
- Léger (~90 Mo), déjà téléchargé chez toi.
- Limite connue : parfois moins bon que l'hybride sur du jargon 10-K
  (à documenter dans l'éval / le README CV).
- BGE-reranker-base reporté (trop lourd pour le disque actuel ~1.1 Go).

Usage :
    python -m src.retrieval.rerank_search \\
      --query "What are Apple's supply chain risks?" \\
      --company Apple \\
      --compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from src.embeddings.embed_chunks import DEFAULT_MODEL_NAME
from src.indexing.index_qdrant import (
    COLLECTION_NAME,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_QDRANT_PATH,
)
from src.retrieval.hybrid_search import (
    DEFAULT_CANDIDATES,
    BM25Index,
    SearchHit,
    _print_hits,
    hybrid_search,
    load_corpus,
)

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(
    query: str,
    hits: list[SearchHit],
    model: CrossEncoder,
    top_k: int = 5,
) -> list[SearchHit]:
    """
    Reclasse `hits` avec un cross-encoder.

    predict() attend une liste de paires (query, document).
    Plus le score est élevé, plus le chunk est jugé pertinent.
    """
    if not hits:
        return []

    pairs = [(query, hit.text) for hit in hits]
    scores = model.predict(pairs)

    rescored: list[SearchHit] = []
    for hit, score in zip(hits, scores):
        rescored.append(
            SearchHit(
                chunk_id=hit.chunk_id,
                text=hit.text,
                company=hit.company,
                section_title=hit.section_title,
                score=float(score),
                source="rerank",
            )
        )

    rescored.sort(key=lambda h: h.score, reverse=True)
    return rescored[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recherche hybride + reranking cross-encoder."
    )
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--company", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
        help="Taille du pool hybride avant rerank (ex. 20)",
    )
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    parser.add_argument("--reranker", type=str, default=DEFAULT_RERANKER)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Affiche hybride (avant) puis rerank (après)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Chargement corpus BM25…")
    corpus = load_corpus(args.embeddings_dir)
    bm25_index = BM25Index(corpus)

    print(f"Chargement embedder {DEFAULT_MODEL_NAME}…")
    embedder = SentenceTransformer(DEFAULT_MODEL_NAME)

    print(f"Chargement reranker {args.reranker}…")
    reranker = CrossEncoder(args.reranker)

    # hybrid_search renvoie déjà hybrid tronqué à top_k — on veut le pool large.
    # Astuce : demander top_k=candidates pour récupérer tout le pool RRF.
    _, _, hybrid_pool = hybrid_search(
        args.query,
        model=embedder,
        bm25_index=bm25_index,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        candidates=args.candidates,
        top_k=args.candidates,
        company=args.company,
    )

    reranked = rerank(args.query, hybrid_pool, reranker, top_k=args.top_k)

    print(f"\nQuery: {args.query!r}")
    if args.company:
        print(f"Filtre company={args.company!r}")

    if args.compare:
        _print_hits("HYBRIDE avant rerank", hybrid_pool, args.top_k)

    _print_hits("APRÈS RERANK", reranked, args.top_k)


if __name__ == "__main__":
    main()
