"""
Recherche hybride dense (Qdrant) + lexicale (BM25) + fusion RRF (brique 5).

Pourquoi cette brique :
- Dense capture le sens, BM25 les mots exacts.
- Ensemble (via RRF), on réduit les faux amis du type "supply agreements"
  quand on cherche "supply chain risks".

Usage (venv activé) :
    pip install -r requirements.txt
    python -m src.retrieval.hybrid_search \\
      --query "What are Apple's supply chain risks?" \\
      --company Apple \\
      --compare
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.embeddings.embed_chunks import DEFAULT_MODEL_NAME, embed_query
from src.indexing.index_qdrant import (
    COLLECTION_NAME,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_QDRANT_PATH,
    get_client,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Combien de candidats on tire de CHAQUE branche avant fusion.
DEFAULT_CANDIDATES = 20
# Paramètre classique RRF (papier Cormack et al.) — ne pas le confondre
# avec le top-k final affiché à l'utilisateur.
RRF_K = 60


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    company: str
    section_title: str
    score: float
    source: str  # "dense" | "bm25" | "hybrid"


def tokenize(text: str) -> list[str]:
    """
    Tokenisation volontairement simple (minuscules + alphanum).

    Piège : on ignore la lemmatisation ("risks" ≠ "risk"). Suffisant en V1 ;
    un vrai moteur (Elastic) ferait stemming / analyzeurs plus riches.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def load_corpus(embeddings_dir: Path) -> list[dict]:
    """Charge les chunks (texte + meta). Les vecteurs ne sont pas nécessaires pour BM25."""
    records: list[dict] = []
    files = sorted(embeddings_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Aucun JSONL dans {embeddings_dir}")
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                records.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "text": row["text"],
                        "company": row["company"],
                        "section_title": row["section_title"],
                    }
                )
    return records


class BM25Index:
    """Index BM25 en mémoire — OK pour ~2k chunks GAFAM."""

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self._tokens = [tokenize(doc["text"]) for doc in corpus]
        self._bm25 = BM25Okapi(self._tokens)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_CANDIDATES,
        company: str | None = None,
    ) -> list[SearchHit]:
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Filtre company AVANT de prendre le top-k (sinon pollution GAFAM).
        candidates: list[tuple[int, float]] = []
        for idx, score in enumerate(scores):
            if company and self.corpus[idx]["company"] != company:
                continue
            candidates.append((idx, float(score)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        hits: list[SearchHit] = []
        for idx, score in candidates[:top_k]:
            doc = self.corpus[idx]
            hits.append(
                SearchHit(
                    chunk_id=doc["chunk_id"],
                    text=doc["text"],
                    company=doc["company"],
                    section_title=doc["section_title"],
                    score=score,
                    source="bm25",
                )
            )
        return hits


def dense_search(
    query: str,
    model: SentenceTransformer,
    qdrant_path: Path,
    collection_name: str = COLLECTION_NAME,
    top_k: int = DEFAULT_CANDIDATES,
    company: str | None = None,
) -> list[SearchHit]:
    """Branche dense : même embedding query (préfixe BGE) que la brique 4."""
    query_vector = embed_query(query, model)
    query_filter = None
    if company:
        query_filter = Filter(
            must=[FieldCondition(key="company", match=MatchValue(value=company))]
        )

    client = get_client(qdrant_path)
    try:
        points = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points
    finally:
        client.close()

    hits: list[SearchHit] = []
    for point in points:
        payload = point.payload or {}
        hits.append(
            SearchHit(
                chunk_id=payload.get("chunk_id", str(point.id)),
                text=payload.get("text", ""),
                company=payload.get("company", ""),
                section_title=payload.get("section_title", ""),
                score=float(point.score or 0.0),
                source="dense",
            )
        )
    return hits


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]],
    rrf_k: int = RRF_K,
) -> list[SearchHit]:
    """
    Fusionne plusieurs classements via RRF.

    On additionne 1/(rrf_k + rang) pour chaque liste où le chunk apparaît.
    Les scores bruts dense/BM25 ne sont PAS mélangés (échelles incompatibles).
    """
    scores: dict[str, float] = defaultdict(float)
    meta: dict[str, SearchHit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.chunk_id] += 1.0 / (rrf_k + rank + 1)
            # Garde le payload du premier aperçu rencontré.
            if hit.chunk_id not in meta:
                meta[hit.chunk_id] = hit

    fused: list[SearchHit] = []
    for chunk_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        base = meta[chunk_id]
        fused.append(
            SearchHit(
                chunk_id=base.chunk_id,
                text=base.text,
                company=base.company,
                section_title=base.section_title,
                score=score,
                source="hybrid",
            )
        )
    return fused


def hybrid_search(
    query: str,
    *,
    model: SentenceTransformer,
    bm25_index: BM25Index,
    qdrant_path: Path = DEFAULT_QDRANT_PATH,
    collection_name: str = COLLECTION_NAME,
    candidates: int = DEFAULT_CANDIDATES,
    top_k: int = 5,
    company: str | None = None,
) -> tuple[list[SearchHit], list[SearchHit], list[SearchHit]]:
    """
    Retourne (dense_hits, bm25_hits, hybrid_hits).

    Les deux premières listes sont tronquées à `candidates` ;
    la liste hybride est tronquée à `top_k` après RRF.
    """
    dense_hits = dense_search(
        query,
        model=model,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        top_k=candidates,
        company=company,
    )
    bm25_hits = bm25_index.search(query, top_k=candidates, company=company)
    hybrid_hits = reciprocal_rank_fusion([dense_hits, bm25_hits])[:top_k]
    return dense_hits, bm25_hits, hybrid_hits


def _print_hits(title: str, hits: list[SearchHit], limit: int) -> None:
    print(f"\n=== {title} (top {limit}) ===")
    for i, hit in enumerate(hits[:limit], start=1):
        preview = hit.text.replace("\n", " ")[:220]
        print(
            f"{i}. score={hit.score:.4f} | {hit.company} | {hit.section_title}"
        )
        print(f"   {preview}…\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recherche hybride dense + BM25.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--company", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Affiche dense seul, BM25 seul, puis hybride (utile pour comprendre)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Chargement corpus BM25…")
    corpus = load_corpus(args.embeddings_dir)
    bm25_index = BM25Index(corpus)
    print(f"  {len(corpus)} chunks indexés en BM25")

    print(f"Chargement modèle {DEFAULT_MODEL_NAME}…")
    model = SentenceTransformer(DEFAULT_MODEL_NAME)

    dense_hits, bm25_hits, hybrid_hits = hybrid_search(
        args.query,
        model=model,
        bm25_index=bm25_index,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        candidates=args.candidates,
        top_k=args.top_k,
        company=args.company,
    )

    print(f"\nQuery: {args.query!r}")
    if args.company:
        print(f"Filtre company={args.company!r}")

    if args.compare:
        _print_hits("DENSE seul", dense_hits, args.top_k)
        _print_hits("BM25 seul", bm25_hits, args.top_k)

    _print_hits("HYBRIDE (RRF)", hybrid_hits, args.top_k)


if __name__ == "__main__":
    main()
