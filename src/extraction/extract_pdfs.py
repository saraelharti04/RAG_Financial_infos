"""
Convertit les PDF 10-K en Markdown avec Docling.

Pourquoi cette brique existe :
- Un PDF n'est pas du texte propre (colonnes, en-têtes, tableaux).
- Docling reconstruit une structure (titres, paragraphes, tableaux) exportable en Markdown,
  format lisible pour vérifier la qualité avant le chunking.

Usage (depuis la racine du projet, venv activé) :
    python -m src.extraction.extract_pdfs --only apple
    python -m src.extraction.extract_pdfs
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

# Chemins relatifs à la racine du repo — on évite les chemins absolus pour que ça marche
# sur une autre machine (CV / clone Git).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "10-K"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "markdown"

# Mapping nom de fichier → nom d'entreprise (métadonnée en tête du .md).
# Si tu renommes un PDF, ajoute une entrée ici.
COMPANY_FROM_STEM = {
    "10-K_amazon": "Amazon",
    "10-K_apple": "Apple",
    "10-K_google": "Alphabet",
    "10-K_meta": "Meta",
    "10-K_microsoft": "Microsoft",
}


def company_name_from_pdf(pdf_path: Path) -> str:
    """Dérive un nom d'entreprise à partir du nom de fichier."""
    return COMPANY_FROM_STEM.get(pdf_path.stem, pdf_path.stem)


def build_markdown_header(company: str, source_pdf: Path) -> str:
    """
    Petit en-tête en haut du fichier Markdown.

    Ce n'est pas du JSON : juste des infos stables pour le debug et, plus tard,
    pour attacher des métadonnées aux chunks (entreprise, source).
    """
    return (
        f"<!--\n"
        f"company: {company}\n"
        f"source_pdf: {source_pdf.name}\n"
        f"extractor: docling\n"
        f"-->\n\n"
        f"# {company} — Form 10-K\n\n"
        f"*Source : `{source_pdf.name}`*\n\n"
        f"---\n\n"
    )


def convert_pdf_to_markdown(pdf_path: Path, converter: DocumentConverter) -> str:
    """
    Lance Docling sur un PDF et renvoie le Markdown du document.

    Piège connu : sur un 10-K (~80–150 pages), la conversion peut prendre plusieurs
    minutes la première fois (téléchargement des modèles + analyse layout/tableaux).
    """
    result = converter.convert(str(pdf_path))
    # export_to_markdown : texte + titres ; images remplacées par un placeholder
    # (on n'en a pas besoin pour un RAG texte en V1).
    return result.document.export_to_markdown()


def extract_one(pdf_path: Path, output_dir: Path, converter: DocumentConverter) -> Path:
    """Extrait un PDF et écrit le .md correspondant dans output_dir."""
    company = company_name_from_pdf(pdf_path)
    output_path = output_dir / f"{pdf_path.stem}.md"

    print(f"→ Extraction : {pdf_path.name} ({company})")
    started = time.perf_counter()
    body = convert_pdf_to_markdown(pdf_path, converter)
    elapsed = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown_header(company, pdf_path) + body, encoding="utf-8")

    # Stats rapides pour juger si l'extraction a "l'air" raisonnable
    n_chars = len(body)
    n_headings = len(re.findall(r"(?m)^#{1,6} ", body))
    print(
        f"  OK en {elapsed:.1f}s → {output_path.relative_to(PROJECT_ROOT)} "
        f"({n_chars:,} caractères, ~{n_headings} titres Markdown)"
    )
    return output_path


def resolve_pdfs(input_dir: Path, only: str | None) -> list[Path]:
    """Liste les PDF à traiter ; --only filtre par sous-chaîne (ex. 'apple')."""
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {input_dir}")

    if only:
        needle = only.lower()
        pdfs = [p for p in pdfs if needle in p.stem.lower()]
        if not pdfs:
            raise FileNotFoundError(
                f"Aucun PDF ne correspond à --only {only!r} dans {input_dir}"
            )
    return pdfs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrait les 10-K PDF en Markdown via Docling."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Dossier des PDF (défaut : data/10-K)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier de sortie Markdown (défaut : data/processed/markdown)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Ne traiter qu'un fichier (sous-chaîne du nom, ex. apple)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdfs = resolve_pdfs(args.input_dir, args.only)

    # Un seul converter réutilisé : évite de recharger les modèles à chaque PDF.
    converter = DocumentConverter()

    print(f"{len(pdfs)} PDF à extraire → {args.output_dir}")
    for pdf in pdfs:
        extract_one(pdf, args.output_dir, converter)
    print("Terminé.")


if __name__ == "__main__":
    main()
