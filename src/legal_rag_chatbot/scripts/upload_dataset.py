from pathlib import Path

from legal_rag_chatbot.ingestion.prepare import build_embedding_file
from legal_rag_chatbot.ingestion.upload import upload_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_legal_counsel_documents.jsonl"
OCR_DIR = PROJECT_ROOT / "data" / "preprocessed"
EMBEDDING_PATH = (
    PROJECT_ROOT / "data" / "chunks" / "synthetic_legal_counsel_documents_chunks.jsonl"
)


BUCKET_NAME = "legal-rag-chatbot-documents"
DESTINATION_PATH = "raw/synthetic_legal_counsel_documents_chunks.jsonl"


def main() -> None:
    """Chunk the OCR results and upload the chunk file to Google Cloud Storage."""
    count = build_embedding_file(DATASET_PATH, OCR_DIR, EMBEDDING_PATH)
    print(f"Wrote {count} chunks to {EMBEDDING_PATH}")

    upload_file(
        bucket_name=BUCKET_NAME,
        local_path=EMBEDDING_PATH,
        destination_path=DESTINATION_PATH,
    )


if __name__ == "__main__":
    main()
