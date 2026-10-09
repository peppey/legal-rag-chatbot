from pathlib import Path

from legal_rag_chatbot.ingestion.prepare import build_embedding_file
from legal_rag_chatbot.ingestion.upload import upload_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "synthetic_legal_counsel_documents.jsonl"
)
EMBEDDING_PATH = (
    PROJECT_ROOT / "data" / "chunks" / "synthetic_legal_counsel_documents_slim.jsonl"
)


BUCKET_NAME = "legal-rag-chatbot-documents"
DESTINATION_PATH = "raw/synthetic_legal_counsel_documents_slim.jsonl"


def main() -> None:
    """Build the slim embedding dataset and upload it to Google Cloud Storage."""
    count = build_embedding_file(DATASET_PATH, EMBEDDING_PATH)
    print(f"Wrote {count} documents to {EMBEDDING_PATH}")

    upload_file(
        bucket_name=BUCKET_NAME,
        local_path=EMBEDDING_PATH,
        destination_path=DESTINATION_PATH,
    )


if __name__ == "__main__":
    main()
