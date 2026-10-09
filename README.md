# RAG hybride sur les 10-K GAFAM

Système de question-réponse ancré dans des rapports financiers SEC (Form 10-K). La V1 couvre le texte narratif : extraction, recherche hybride, génération locale et interface de démo. Les chiffres issus des tableaux sont prévus en V2.

Auteure : Sara El Harti — Master Data Science × Financial Engineering.

One-pagers : [`docs/one-pager-RAG-10K.html`](docs/one-pager-RAG-10K.html) (FR), [`docs/one-pager-RAG-10K-EN.html`](docs/one-pager-RAG-10K-EN.html) (EN).

## Corpus

Cinq 10-K les plus récents : Apple, Amazon, Alphabet, Meta, Microsoft. Les clôtures fiscales ne tombent pas le même jour (Apple en septembre, Microsoft en juin, les autres en décembre).

Les PDF ne sont pas dans le dépôt. Place-les dans `data/10-K/` :

```
data/10-K/10-K_apple.pdf
data/10-K/10-K_amazon.pdf
data/10-K/10-K_google.pdf
data/10-K/10-K_meta.pdf
data/10-K/10-K_microsoft.pdf
```

## V1 — pipeline

1. **Extraction** — Docling convertit chaque PDF en Markdown (`data/processed/markdown/`).
2. **Chunking hybride** — découpe sur les titres Markdown, puis par taille avec overlap si une section est trop longue (`data/processed/chunks/`).
3. **Embeddings** — `BAAI/bge-small-en-v1.5` (384 dimensions, vecteurs normalisés). Les passages sont encodés tels quels. Les questions portent le préfixe BGE.
4. **Vector store** — Qdrant local, distance cosinus, filtre par entreprise (`data/qdrant/`).
5. **Recherche hybride** — similarité dense + BM25, fusion par Reciprocal Rank Fusion (constante 60).
6. **Rerank** — cross-encoder MiniLM, optionnel. Sur le test « supply chain » Apple, l’hybride seul classait mieux.
7. **Génération** — Ollama `qwen2.5:3b` par défaut. Le prompt impose de répondre uniquement à partir des extraits, de citer la section, et de ne pas inventer de chiffres. Claude reste disponible avec `--llm claude`.
8. **Démo** — Streamlit.
9. **Évaluation** — petit jeu de questions dans `data/eval/questions.json` et métriques RAGAS (juge local, scores indicatifs).

## Stack

Python, Docling, sentence-transformers, Qdrant, rank-bm25, Ollama, Streamlit, RAGAS.

## Lancer la démo

Prérequis : Python 3.9+, [Ollama](https://ollama.com) avec le modèle `qwen2.5:3b`, et les PDF déjà extraits, chunkés, embeddés et indexés (commandes plus bas).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama serve
# dans un autre terminal, une fois l'index construit :
.venv/bin/streamlit run app/streamlit_app.py
```

Question de démo, filtre Apple, rerank désactivé : `What are Apple's supply chain risks?`

## Reconstruire l’index

Depuis la racine, venv activé. La première extraction Docling télécharge des modèles et prend plusieurs minutes par PDF.

```bash
python -m src.extraction.extract_pdfs
python -m src.chunking.chunk_markdown
python -m src.embeddings.embed_chunks
python -m src.indexing.index_qdrant
```

Recherche en ligne de commande :

```bash
python -m src.generation.rag_answer \
  --query "What are Apple's supply chain risks?" \
  --company Apple
```

Qdrant en mode dossier local n’accepte qu’un seul processus à la fois. Ferme Streamlit avant de ré-indexer.

## Ce qui n’est pas versionné

Le `.gitignore` exclut l’environnement virtuel, les PDF, les sorties régénérables (`data/processed/`), la base Qdrant et le fichier `.env`. Un modèle d’environnement est dans `.env.example` (clé Anthropic, seulement si tu passes sur Claude).

## Roadmap

- **V2** — postes financiers normalisés dans DuckDB, questions chiffrées via text-to-SQL. Le chiffre affiché doit venir du résultat SQL, pas du modèle.
- **V3** — function calling : le système choisit le texte, le SQL, ou les deux, puis fusionne les réponses.

## Limites connues

L’extraction PDF coupe parfois des mots et abîme les tableaux. Le chunking par caractères peut reprendre au milieu d’un mot. La recherche dense rapproche aussi des faux amis (« supply agreements » n’est pas un risque de chaîne logistique) : c’est le rôle de BM25 et de la fusion. Qwen 2.5 3B suit le prompt mais peut ajouter un fait absent des extraits. Le juge RAGAS est le même petit modèle, donc les scores servent à itérer, pas comme benchmark.
