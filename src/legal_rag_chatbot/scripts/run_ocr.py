import argparse
from pathlib import Path

from legal_rag_chatbot.ingestion.ocr import run_ocr
from legal_rag_chatbot.rag.query import PROJECT_ID

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed"

# Document AI is not available in europe-west3; "eu" keeps the data in the EU.
LOCATION = "eu"
PROCESSOR_DISPLAY_NAME = "legal-rag-ocr"


def main() -> None:
    """OCR the PDFs in data/raw/pdf and write one JSON file per PDF to data/preprocessed."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--force", action="store_true", help="Redo existing results")
    parser.add_argument("--limit", type=int, help="Only process the first N PDFs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    count = run_ocr(
        PDF_DIR,
        OUTPUT_DIR,
        PROJECT_ID,
        LOCATION,
        PROCESSOR_DISPLAY_NAME,
        workers=args.workers,
        force=args.force,
        limit=args.limit,
    )
    print(f"Wrote {count} OCR results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
