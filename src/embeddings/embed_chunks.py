"""
Embeddings des chunks JSONL avec BAAI/bge-small-en-v1.5 (brique 3).

Pourquoi cette brique :
- Chaque chunk devient un vecteur (ici dim=384) pour mesurer la similarité
  sémantique entre une question et les passages du 10-K.
- Sans embeddings, Qdrant ne peut pas faire de recherche "dense".

Piège BGE (important pour plus tard) :
- Les DOCUMENTS (chunks) s'encodent tels quels.
- Les QUESTIONS doivent être préfixées (voir QUERY_PREFIX) — sinon la
  qualité de retrieval baisse. On expose embed_query() pour la brique recherche.

Usage (venv activé) :
    pip install -r requirements.txt   # si pas encore fait pour cette brique
    python -m src.embeddings.embed_chunks --only apple
    python -m src.embeddings.embed_chunks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"

# Modèle choisi ensemble : bon compromis qualité / poids pour un Mac.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Instruction officielle BGE v1.5 pour les requêtes (pas pour les passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks(jsonl_path: Path) -> list[dict]:
    """Charge un fichier .jsonl de chunks (1 objet JSON par ligne)."""
    chunks: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_passages(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Encode une liste de passages (chunks).

    normalize_embeddings=True : vecteurs de norme 1 → la similarité cosinus
    se calcule simplement via un produit scalaire (pratique avec Qdrant).
    """
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return [vec.tolist() for vec in vectors]


def embed_query(query: str, model: SentenceTransformer) -> list[float]:
    """
    Encode une question utilisateur (avec le préfixe BGE).

    On ne l'utilise pas encore dans ce script, mais elle servira pour
    la recherche dans Qdrant — même modèle, préfixe différent.
    """
    vector = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    )
    return vector.tolist()


def embed_file(
    jsonl_path: Path,
    output_dir: Path,
    model: SentenceTransformer,
    batch_size: int,
    model_name: str,
) -> Path:
    """Lit des chunks, ajoute le champ embedding, écrit un .jsonl enrichi."""
    chunks = load_chunks(jsonl_path)
    if not chunks:
        raise ValueError(f"Aucun chunk dans {jsonl_path}")

    texts = [c["text"] for c in chunks]
    print(f"→ {jsonl_path.name}: {len(texts)} chunks à embedder…")
    vectors = embed_passages(texts, model, batch_size=batch_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / jsonl_path.name
    with out_path.open("w", encoding="utf-8") as f:
        for chunk, vector in zip(chunks, vectors):
            record = {
                **chunk,
                "embedding": vector,
                "embedding_model": model_name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    dim = len(vectors[0])
    print(
        f"  OK → {out_path.relative_to(PROJECT_ROOT)} "
        f"({len(vectors)} vecteurs, dim={dim})"
    )
    return out_path


def resolve_chunk_files(input_dir: Path, only: str | None) -> list[Path]:
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Aucun JSONL de chunks dans {input_dir}")
    if only:
        needle = only.lower()
        files = [p for p in files if needle in p.stem.lower()]
        if not files:
            raise FileNotFoundError(f"Aucun fichier pour --only {only!r}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedde les chunks JSONL avec BGE-small."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Nom Hugging Face du modèle (défaut : BGE-small)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = resolve_chunk_files(args.input_dir, args.only)

    # Premier lancement : télécharge le modèle (~130 Mo) dans le cache HF.
    print(f"Chargement du modèle {args.model}…")
    model = SentenceTransformer(args.model)

    print(f"{len(files)} fichier(s) → {args.output_dir}")
    for path in files:
        embed_file(path, args.output_dir, model, args.batch_size, args.model)
    print("Terminé.")


if __name__ == "__main__":
    main()
