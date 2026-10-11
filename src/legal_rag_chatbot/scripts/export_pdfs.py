from pathlib import Path

from legal_rag_chatbot.ingestion.pdf import build_pdfs

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "synthetic_legal_counsel_documents.jsonl"
)
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"


def main() -> None:
    """Render every dataset document as a PDF in data/raw/pdf."""
    count = build_pdfs(DATASET_PATH, PDF_DIR)
    print(f"Wrote {count} PDFs to {PDF_DIR}")


if __name__ == "__main__":
    main()
