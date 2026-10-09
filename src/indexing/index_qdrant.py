"""
Indexation des embeddings dans Qdrant (brique 4).

Pourquoi cette brique :
- Les JSONL d'embeddings ne permettent pas une recherche efficace.
- Qdrant stocke vecteur + payload (texte, company, section…) et renvoie
  les k plus proches voisins d'une question.

Mode choisi : client local persistant (path=...), sans Docker.
Piège : un seul process à la fois sur ce dossier — toujours fermer le client.

Usage (venv activé) :
    pip install -r requirements.txt
    python -m src.indexing.index_qdrant
    python -m src.indexing.index_qdrant --query "What are Apple's supply chain risks?"
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.embeddings.embed_chunks import (
    DEFAULT_MODEL_NAME,
    embed_query,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
COLLECTION_NAME = "ten_k_chunks"
VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5
UPSERT_BATCH_SIZE = 64


def stable_point_id(chunk_id: str) -> str:
    """
    Qdrant n'accepte pas nos chunk_id string libres comme id "brut".
    UUID5 = id déterministe : ré-indexer le même chunk écrase le même point.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def load_embedded_chunks(embeddings_dir: Path) -> list[dict]:
    records: list[dict] = []
    files = sorted(embeddings_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Aucun embedding JSONL dans {embeddings_dir}")
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def get_client(qdrant_path: Path) -> QdrantClient:
    """Ouvre (ou crée) la base locale sur disque."""
    qdrant_path.mkdir(parents=True, exist_ok=True)
    # force_disable_check_same_thread : évite des erreurs sur certains usages notebook
    return QdrantClient(path=str(qdrant_path))


def recreate_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Recrée la collection from scratch.

    Pourquoi recreate plutôt qu'append : en V1 on ré-indexe souvent après
    un nouveau chunking ; plus simple et sans doublons.
    """
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def records_to_points(records: list[dict]) -> list[PointStruct]:
    points: list[PointStruct] = []
    for record in records:
        embedding = record["embedding"]
        if len(embedding) != VECTOR_SIZE:
            raise ValueError(
                f"{record.get('chunk_id')}: dim={len(embedding)}, attendu {VECTOR_SIZE}"
            )
        # Le vecteur vit dans `vector` ; le payload = ce qu'on affichera / filtrera.
        points.append(
            PointStruct(
                id=stable_point_id(record["chunk_id"]),
                vector=embedding,
                payload={
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "company": record["company"],
                    "source_md": record["source_md"],
                    "section_title": record["section_title"],
                    "chunk_index": record["chunk_index"],
                    "embedding_model": record.get(
                        "embedding_model", DEFAULT_MODEL_NAME
                    ),
                },
            )
        )
    return points


def upsert_in_batches(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct],
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """Upsert par lots pour éviter de tout charger d'un coup en mémoire API."""
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"  upsert {min(i + batch_size, len(points))}/{len(points)}")


def index_all(
    embeddings_dir: Path,
    qdrant_path: Path,
    collection_name: str = COLLECTION_NAME,
) -> int:
    records = load_embedded_chunks(embeddings_dir)
    print(f"{len(records)} chunks à indexer dans '{collection_name}'")

    client = get_client(qdrant_path)
    try:
        recreate_collection(client, collection_name)
        points = records_to_points(records)
        upsert_in_batches(client, collection_name, points)
        info = client.get_collection(collection_name)
        count = info.points_count
        print(f"Collection prête : {count} points (dim={VECTOR_SIZE}, cosine)")
        return count or 0
    finally:
        # Important en mode path= : libère le lock sur le dossier.
        client.close()


def search_demo(
    query: str,
    qdrant_path: Path,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 5,
    company: str | None = None,
) -> None:
    """Petit test manuel : embed la question + top-k dans Qdrant."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    print(f"Chargement du modèle {DEFAULT_MODEL_NAME} pour la query…")
    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    query_vector = embed_query(query, model)

    client = get_client(qdrant_path)
    try:
        query_filter = None
        if company:
            # Filtre payload : ex. uniquement les chunks Apple.
            query_filter = Filter(
                must=[FieldCondition(key="company", match=MatchValue(value=company))]
            )

        hits = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        print(f"\nQuery: {query!r}")
        if company:
            print(f"Filtre company={company!r}")
        print(f"Top {len(hits)} :\n")
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            preview = (payload.get("text") or "").replace("\n", " ")[:220]
            print(
                f"{rank}. score={hit.score:.4f} | {payload.get('company')} | "
                f"{payload.get('section_title')}"
            )
            print(f"   {preview}…\n")
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indexe les embeddings dans Qdrant.")
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Si fourni : skip l'indexation et lance une recherche de démo",
    )
    parser.add_argument("--company", type=str, default=None, help="Filtre optionnel")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force une (re)indexation même si --query est fourni",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.query and not args.reindex:
        search_demo(
            query=args.query,
            qdrant_path=args.qdrant_path,
            collection_name=args.collection,
            top_k=args.top_k,
            company=args.company,
        )
        return

    index_all(args.embeddings_dir, args.qdrant_path, args.collection)

    if args.query:
        search_demo(
            query=args.query,
            qdrant_path=args.qdrant_path,
            collection_name=args.collection,
            top_k=args.top_k,
            company=args.company,
        )


if __name__ == "__main__":
    main()
