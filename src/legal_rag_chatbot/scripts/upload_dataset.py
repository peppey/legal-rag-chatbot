from pathlib import Path

from legal_rag_chatbot.access import GROUPS, OCR_DIR, chunk_file_name
from legal_rag_chatbot.ingestion.prepare import build_embedding_files
from legal_rag_chatbot.ingestion.upload import upload_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_legal_counsel_documents.jsonl"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"


BUCKET_NAME = "legal-rag-chatbot-documents"
DESTINATION_DIR = "raw"


def main() -> None:
    """Chunk the OCR results per access group and upload the chunk files to Google Cloud Storage."""
    counts = build_embedding_files(DATASET_PATH, OCR_DIR, CHUNKS_DIR)
    for group in GROUPS:
        print(f"Wrote {counts[group]} chunks for group {group}")
        upload_file(
            bucket_name=BUCKET_NAME,
            local_path=CHUNKS_DIR / chunk_file_name(group),
            destination_path=f"{DESTINATION_DIR}/{chunk_file_name(group)}",
        )


if __name__ == "__main__":
    main()
