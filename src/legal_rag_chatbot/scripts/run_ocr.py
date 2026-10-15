import argparse

from legal_rag_chatbot.access import OCR_DIR as OUTPUT_DIR
from legal_rag_chatbot.access import PDF_DIR
from legal_rag_chatbot.ingestion.ocr import (
    OCR_LOCATION,
    OCR_PROCESSOR_NAME,
    run_ocr,
)
from legal_rag_chatbot.rag.query import PROJECT_ID


def main() -> None:
    """OCR the PDFs in data/raw/pdf_with_noise/{A,B,Geteilt} and write one JSON file per PDF to data/preprocessed."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--force", action="store_true", help="Redo existing results")
    parser.add_argument("--limit", type=int, help="Only process the first N PDFs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    count = run_ocr(
        PDF_DIR,
        OUTPUT_DIR,
        PROJECT_ID,
        OCR_LOCATION,
        OCR_PROCESSOR_NAME,
        workers=args.workers,
        force=args.force,
        limit=args.limit,
    )
    print(f"Wrote {count} OCR results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
