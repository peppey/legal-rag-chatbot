from pathlib import Path

from legal_rag_chatbot.ingestion.upload import upload_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "synthetic_legal_counsel_documents.jsonl"
)


BUCKET_NAME = "legal-rag-chatbot-documents"
DESTINATION_PATH = "raw/synthetic_legal_counsel_documents.jsonl"


def main() -> None:
    """Upload the synthetic legal document dataset to Google Cloud Storage."""
    upload_file(
        bucket_name=BUCKET_NAME,
        local_path=DATASET_PATH,
        destination_path=DESTINATION_PATH,
    )


if __name__ == "__main__":
    main()
