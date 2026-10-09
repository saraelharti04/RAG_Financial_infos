"""
Chunking hybride des Markdown 10-K (brique 2).

Stratégie validée :
1) découper selon la structure (titres Markdown # / ## / ### …)
2) si une section dépasse max_chars → re-couper avec overlap

Usage (venv activé, depuis la racine du projet) :
    python -m src.chunking.chunk_markdown --only apple
    python -m src.chunking.chunk_markdown
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "markdown"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"

# Taille max d'un chunk en caractères (approx.). On utilise des caractères
# plutôt que des tokens pour rester simple en V1 — ce n'est pas exact à 100 %,
# mais largement suffisant pour démarrer.
DEFAULT_MAX_CHARS = 2000
# Chevauchement entre deux sous-chunks d'une même longue section.
DEFAULT_OVERLAP_CHARS = 200

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    """Un morceau de document prêt pour l'embedding plus tard."""

    chunk_id: str
    text: str
    company: str
    source_md: str
    section_title: str
    chunk_index: int


def parse_company_from_header(markdown: str, fallback: str) -> str:
    """Lit la méta `company:` laissée par l'extraction Docling."""
    match = re.search(r"^company:\s*(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else fallback


# ---------------------------------------------------------------------------
# TODO 1 — Découpage structurel
# ---------------------------------------------------------------------------
def split_by_headings(markdown: str) -> list[tuple[str, str]]:
    """
    Découpe le Markdown en sections (titre, corps).

    Attendu :
    - une liste de tuples (section_title, section_text)
    - section_text contient le titre + le paragraphe qui suit, jusqu'au
      prochain titre de n'importe quel niveau
    - s'il y a du texte AVANT le premier titre, utilise
      section_title = "Preamble"

    Indice :
    - `HEADING_RE.finditer(markdown)` te donne la position de chaque titre
    - pour chaque titre i, le corps va de start(i) jusqu'à start(i+1)
      (ou la fin du document pour le dernier)

    Piège : ne renvoie pas de sections vides (titre sans contenu utile).
    """
    matches = list(HEADING_RE.finditer(markdown))
    sections: list[tuple[str, str]] = []

    # Texte avant le premier titre (commentaire HTML, etc.)
    if matches:
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            sections.append(("Preamble", preamble))
    else:
        # Aucun titre : tout le document est une seule section
        body = markdown.strip()
        if body:
            sections.append(("Preamble", body))
        return sections

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        section_text = markdown[start:end].strip()
        if not section_text:
            continue
        # group(2) = libellé du titre sans les "#"
        section_title = match.group(2).strip() or "Untitled"
        sections.append((section_title, section_text))

    return sections


# ---------------------------------------------------------------------------
# TODO 2 — Découpage par taille + overlap
# ---------------------------------------------------------------------------
def split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """
    Découpe `text` en morceaux d'au plus max_chars, avec chevauchement.

    Règles :
    - si len(text) <= max_chars → renvoyer [text]
    - sinon, avancer une fenêtre de taille max_chars
    - le début du morceau suivant = fin du précédent - overlap_chars
    - overlap_chars doit être < max_chars (sinon boucle infinie)

    Bonus (si tu veux aller plus loin) :
    - essayer de couper sur un saut de ligne plutôt qu'au milieu d'un mot

    Piège : vérifier que tu progresses bien à chaque itération
    (sinon while True qui ne s'arrête jamais).
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars doit être strictement < max_chars")

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)

        # Si on n'est pas à la fin, tenter une coupe sur un saut de ligne
        # dans la 2e moitié de la fenêtre — évite de couper au milieu d'un mot.
        if end < n:
            window = text[start:end]
            cut = window.rfind("\n", max_chars // 2)
            if cut != -1:
                end = start + cut + 1  # inclut le \n dans le chunk courant

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= n:
            break

        # Avance avec overlap ; max(..., start+1) garantit la progression.
        next_start = end - overlap_chars
        start = max(next_start, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# TODO 3 — Orchestration hybride
# ---------------------------------------------------------------------------
def chunk_document(
    markdown: str,
    *,
    company: str,
    source_md: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """
    Pipeline hybride :
    1. split_by_headings
    2. pour chaque section trop longue → split_with_overlap
    3. fabriquer des objets Chunk avec un chunk_id stable

    Suggestion de chunk_id :
        f"{source_md.replace('.md', '')}::{chunk_index:04d}"

    section_title : le titre de la section d'origine (même si re-coupé).
    chunk_index : 0, 1, 2, … sur tout le document.
    """
    sections = split_by_headings(markdown)
    chunks: list[Chunk] = []
    stem = source_md.replace(".md", "")

    for section_title, section_text in sections:
        # Hybride : structure d'abord, taille ensuite si besoin.
        pieces = split_with_overlap(section_text, max_chars, overlap_chars)
        for piece in pieces:
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}::{idx:04d}",
                    text=piece,
                    company=company,
                    source_md=source_md,
                    section_title=section_title,
                    chunk_index=idx,
                )
            )

    return chunks


def chunk_file(
    md_path: Path,
    output_dir: Path,
    max_chars: int,
    overlap_chars: int,
) -> Path:
    """Lit un .md, écrit un .jsonl (1 chunk = 1 ligne JSON)."""
    markdown = md_path.read_text(encoding="utf-8")
    company = parse_company_from_header(markdown, fallback=md_path.stem)

    chunks = chunk_document(
        markdown,
        company=company,
        source_md=md_path.name,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{md_path.stem}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    avg = (sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0
    print(
        f"→ {md_path.name}: {len(chunks)} chunks "
        f"(taille moyenne ~{avg:.0f} car.) → {out_path.relative_to(PROJECT_ROOT)}"
    )
    return out_path


def resolve_markdown_files(input_dir: Path, only: str | None) -> list[Path]:
    files = sorted(input_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"Aucun Markdown dans {input_dir}")
    if only:
        needle = only.lower()
        files = [p for p in files if needle in p.stem.lower()]
        if not files:
            raise FileNotFoundError(f"Aucun fichier pour --only {only!r}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunking hybride des 10-K Markdown.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overlap_chars >= args.max_chars:
        raise ValueError("overlap_chars doit être strictement < max_chars")

    files = resolve_markdown_files(args.input_dir, args.only)
    print(f"{len(files)} fichier(s) à chunker → {args.output_dir}")
    for path in files:
        chunk_file(path, args.output_dir, args.max_chars, args.overlap_chars)
    print("Terminé.")


if __name__ == "__main__":
    main()
