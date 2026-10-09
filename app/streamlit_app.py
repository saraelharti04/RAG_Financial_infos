"""
Interface Streamlit du RAG 10-K (démo interactive).

Lance depuis la racine du projet (venv activé, Ollama ouvert) :
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

# Garantit que `import src...` marche quand Streamlit lance ce fichier.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embed_chunks import DEFAULT_MODEL_NAME
from src.generation.rag_answer import (
    DEFAULT_OLLAMA_MODEL,
    generate_answer,
    retrieve_hits,
)
from src.indexing.index_qdrant import DEFAULT_EMBEDDINGS_DIR, DEFAULT_QDRANT_PATH
from src.retrieval.hybrid_search import BM25Index, load_corpus

COMPANIES = ["Apple", "Amazon", "Alphabet", "Meta", "Microsoft"]


@st.cache_resource(show_spinner="Chargement BM25 + embeddings (1re fois)…")
def load_retrieval_resources():
    """
    Cache Streamlit : on ne recharge pas le corpus / le modèle à chaque clic.
    (Les poids restent en mémoire tant que l'app tourne.)
    """
    corpus = load_corpus(DEFAULT_EMBEDDINGS_DIR)
    bm25 = BM25Index(corpus)
    # local_files_only : le modèle est déjà dans le cache HF.
    # Sans ça, SentenceTransformer interroge huggingface.co (adapter_config.json)
    # et un proxy qui répond 403 fait échouer toute la requête.
    embedder = SentenceTransformer(DEFAULT_MODEL_NAME, local_files_only=True)
    return bm25, embedder


def run_rag(query: str, company: str | None, top_k: int, use_rerank: bool):
    """Wrapper autour du pipeline CLI, avec ressources Streamlit cachées."""
    bm25, embedder = load_retrieval_resources()
    hits = retrieve_hits(
        query,
        company=company,
        top_k=top_k,
        candidates=20,
        use_rerank=use_rerank,
        embeddings_dir=DEFAULT_EMBEDDINGS_DIR,
        qdrant_path=DEFAULT_QDRANT_PATH,
        collection="ten_k_chunks",
        bm25_index=bm25,
        embedder=embedder,
    )
    answer = generate_answer(
        query,
        hits,
        llm="ollama",
        model=DEFAULT_OLLAMA_MODEL,
    )
    return hits, answer


def main() -> None:
    st.set_page_config(
        page_title="RAG 10-K GAFAM",
        page_icon="📄",
        layout="wide",
    )

    st.title("RAG sur les 10-K GAFAM")
    st.caption(
        "Retrieval hybride (dense + BM25) → réponse locale via Ollama (qwen2.5:3b)"
    )

    with st.sidebar:
        st.header("Paramètres")
        company = st.selectbox(
            "Entreprise",
            options=["Toutes"] + COMPANIES,
            index=1,  # Apple par défaut (notre query de test)
        )
        top_k = st.slider("Nombre de chunks (top-k)", min_value=3, max_value=8, value=5)
        use_rerank = st.checkbox(
            "Activer rerank MiniLM",
            value=False,
            help="Souvent inutile : l'hybride seul était meilleur sur notre test.",
        )
        st.divider()
        st.markdown(
            f"**LLM :** Ollama / `{DEFAULT_OLLAMA_MODEL}`  \n"
            "**Index :** Qdrant local + BM25"
        )
        st.info("Ollama doit être ouvert avant de poser une question.")

    query = st.text_area(
        "Ta question",
        value="What are Apple's supply chain risks?",
        height=100,
    )

    col_btn, _ = st.columns([1, 4])
    asked = col_btn.button("Répondre", type="primary", use_container_width=True)

    if asked:
        if not query.strip():
            st.warning("Écris une question.")
            return

        company_filter = None if company == "Toutes" else company

        with st.spinner("Retrieval + génération Ollama… (peut prendre ~30–90 s)"):
            try:
                hits, answer = run_rag(
                    query.strip(),
                    company_filter,
                    top_k=top_k,
                    use_rerank=use_rerank,
                )
            except Exception as exc:  # affiche l'erreur dans l'UI, pas seulement le terminal
                st.error(f"Erreur : {exc}")
                return

        st.subheader("Réponse")
        st.markdown(answer)

        st.subheader("Chunks utilisés (contexte)")
        for i, hit in enumerate(hits, start=1):
            with st.expander(
                f"{i}. {hit.company} — {hit.section_title} (score={hit.score:.4f})"
            ):
                st.write(hit.text)


if __name__ == "__main__":
    main()
