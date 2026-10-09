"""
Pipeline RAG complet : retrieval → contexte → LLM (brique 7).

Concept :
- On ne demande PAS au LLM de "connaître" les 10-K.
- On lui passe les chunks récupérés (hybride ± rerank) comme CONTEXT.
- Il doit répondre UNIQUEMENT à partir de ce contexte, et dire
  clairement s'il manque d'info (réduit les hallucinations).

Choix V1 :
- Retrieval par défaut = hybride RRF.
- LLM par défaut = Ollama local (qwen2.5:3b) → gratuit, déjà sur ta machine.
- --llm claude reste dispo si tu as une clé Anthropic plus tard.

Prérequis Ollama :
    1. Ollama lancé (l'app Mac ou `ollama serve`)
    2. Modèle présent : ollama list  (tu as déjà qwen2.5:3b)

Usage :
    python -m src.generation.rag_answer \\
      --query "What are Apple's supply chain risks?" \\
      --company Apple

    # Optionnel payant :
    python -m src.generation.rag_answer --llm claude --query "..." --company Apple
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
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
    hybrid_search,
    load_corpus,
)
from src.retrieval.rerank_search import DEFAULT_RERANKER, rerank

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LLM = "ollama"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"


SYSTEM_PROMPT = """Tu es un assistant d'analyse de rapports financiers (Form 10-K SEC).

Règles STRICTES :
1. Réponds UNIQUEMENT à partir du CONTEXT fourni (extraits de 10-K).
2. Si le CONTEXT ne contient pas l'information, dis-le clairement.
   Ne invente pas de chiffres, dates ou faits.
3. Cite tes sources en indiquant [company — section_title] après chaque fait important.
4. Réponds en français, de façon claire et structurée (puces si utile).
5. Reste factuel : tu résumes le document, tu ne donnes pas de conseil d'investissement.
"""


def build_context(hits: list[SearchHit]) -> str:
    """Formate les chunks en bloc CONTEXT lisible pour le LLM."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Extrait {i}]\n"
            f"company: {hit.company}\n"
            f"section: {hit.section_title}\n"
            f"chunk_id: {hit.chunk_id}\n"
            f"texte:\n{hit.text}"
        )
    return "\n\n---\n\n".join(blocks)


def build_user_message(query: str, context: str) -> str:
    return (
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT (extraits 10-K):\n{context}\n\n"
        "Réponds à la QUESTION en respectant les règles."
    )


def retrieve_hits(
    query: str,
    *,
    company: str | None,
    top_k: int,
    candidates: int,
    use_rerank: bool,
    embeddings_dir: Path,
    qdrant_path: Path,
    collection: str,
    bm25_index: BM25Index | None = None,
    embedder: SentenceTransformer | None = None,
) -> list[SearchHit]:
    """Enchaîne BM25 + dense (+ rerank optionnel) et renvoie le top_k final."""
    if bm25_index is None:
        print("Chargement corpus BM25…")
        bm25_index = BM25Index(load_corpus(embeddings_dir))

    if embedder is None:
        print(f"Chargement embedder {DEFAULT_MODEL_NAME}…")
        embedder = SentenceTransformer(DEFAULT_MODEL_NAME)

    pool_size = candidates if use_rerank else top_k
    _, _, hybrid_hits = hybrid_search(
        query,
        model=embedder,
        bm25_index=bm25_index,
        qdrant_path=qdrant_path,
        collection_name=collection,
        candidates=candidates,
        top_k=pool_size,
        company=company,
    )

    if not use_rerank:
        return hybrid_hits[:top_k]

    print(f"Rerank avec {DEFAULT_RERANKER}…")
    reranker = CrossEncoder(DEFAULT_RERANKER)
    return rerank(query, hybrid_hits, reranker, top_k=top_k)


def ask_ollama(
    query: str,
    hits: list[SearchHit],
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """
    Appelle Ollama en local (API /api/chat, sans streaming).

    Aucune clé API : le modèle tourne sur ta machine.
    """
    context = build_context(hits)
    user_message = build_user_message(query, context)
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Impossible de joindre Ollama sur localhost:11434.\n"
            "Vérifie que l'app Ollama est ouverte, ou lance : ollama serve\n"
            f"Détail : {exc}"
        ) from exc

    message = body.get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError(f"Réponse Ollama inattendue : {body!r}")
    return content.strip()


def ask_claude(
    query: str,
    hits: list[SearchHit],
    *,
    model: str = DEFAULT_CLAUDE_MODEL,
    max_tokens: int = 1024,
) -> str:
    """Appelle l'API Claude (optionnel / payant) avec le contexte retrieval."""
    # Import local : pas besoin d'Anthropic si on reste 100 % Ollama.
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY manquante. Pour le mode gratuit, utilise --llm ollama."
        )

    client = Anthropic(api_key=api_key)
    context = build_context(hits)
    user_message = build_user_message(query, context)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def generate_answer(
    query: str,
    hits: list[SearchHit],
    *,
    llm: str,
    model: str | None,
) -> str:
    if llm == "ollama":
        return ask_ollama(query, hits, model=model or DEFAULT_OLLAMA_MODEL)
    if llm == "claude":
        return ask_claude(query, hits, model=model or DEFAULT_CLAUDE_MODEL)
    raise ValueError(f"LLM inconnu : {llm!r} (attendu : ollama | claude)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG : retrieval + réponse LLM (Ollama par défaut)."
    )
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--company", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5, help="Chunks envoyés au LLM")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Active le rerank MiniLM (désactivé par défaut)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["ollama", "claude"],
        default=DEFAULT_LLM,
        help="Backend de génération (défaut : ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nom du modèle (défaut : qwen2.5:3b pour ollama, claude-sonnet-4-6 pour claude)",
    )
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Affiche aussi les chunks utilisés (debug)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    hits = retrieve_hits(
        args.query,
        company=args.company,
        top_k=args.top_k,
        candidates=args.candidates,
        use_rerank=args.rerank,
        embeddings_dir=args.embeddings_dir,
        qdrant_path=args.qdrant_path,
        collection=args.collection,
    )

    model_name = args.model or (
        DEFAULT_OLLAMA_MODEL if args.llm == "ollama" else DEFAULT_CLAUDE_MODEL
    )

    print(f"\nQuery: {args.query!r}")
    if args.company:
        print(f"Filtre company={args.company!r}")
    print(
        f"Retrieval: {'hybride + rerank' if args.rerank else 'hybride RRF'} "
        f"| top_k={len(hits)}"
    )
    print(f"LLM: {args.llm} ({model_name})")

    if args.show_context:
        print("\n--- Chunks envoyés au LLM ---")
        for i, hit in enumerate(hits, start=1):
            preview = hit.text.replace("\n", " ")[:160]
            print(f"{i}. [{hit.company} — {hit.section_title}] {preview}…")

    print(f"\nAppel {args.llm}…")
    answer = generate_answer(args.query, hits, llm=args.llm, model=model_name)

    print("\n=== RÉPONSE ===\n")
    print(answer)


if __name__ == "__main__":
    main()
