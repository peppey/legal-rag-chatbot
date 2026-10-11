import json
from pathlib import Path

from legal_rag_chatbot.ingestion.chunking import chunk_paragraphs, split_at_offsets

# Fields that end up in the embedded chunk text; ids and flags would only dilute it.
EMBEDDED_FIELDS = (
    "document_id",
    "case_id",
    "document_type",
    "document_title",
    "created_at",
    "section",
    "content",
)


def load_ocr(ocr_dir: str | Path, document_id: str) -> dict:
    ocr_file = Path(ocr_dir) / f"{document_id}.json"
    if not ocr_file.is_file():
        raise FileNotFoundError(
            f"No OCR result for {document_id} at {ocr_file}; run the OCR script first"
        )
    return json.loads(ocr_file.read_text(encoding="utf-8"))


def load_documents(source: str | Path, ocr_dir: str | Path) -> list[dict]:
    """Read the raw documents and replace their content with the OCR text of their PDF."""
    with Path(source).open(encoding="utf-8") as src:
        docs = [json.loads(line) for line in src if line.strip()]
    for doc in docs:
        doc["content"] = load_ocr(ocr_dir, doc["document_id"])["text"]
    return docs


def _paragraphs(ocr: dict) -> list[str]:
    try:
        return split_at_offsets(ocr["text"], ocr["paragraph_starts"])
    except KeyError:
        raise ValueError(
            f"OCR result of {ocr['document_id']} has no paragraph offsets; rerun the OCR with --force"
        ) from None


def build_embedding_file(
    source: str | Path, ocr_dir: str | Path, destination: str | Path
) -> int:
    """Write one JSONL line per chunk (document metadata, section, text) and return the chunk count."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with (
        Path(source).open(encoding="utf-8") as src,
        destination.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            doc = json.loads(line)
            ocr = load_ocr(ocr_dir, doc["document_id"])
            chunks = chunk_paragraphs(_paragraphs(ocr))
            if not chunks:
                raise ValueError(
                    f"No text found in the OCR result of {doc['document_id']}"
                )
            for chunk in chunks:
                record = {**doc, **chunk}
                slim = {field: record[field] for field in EMBEDDED_FIELDS}
                dst.write(json.dumps(slim, ensure_ascii=False) + "\n")
                count += 1
    return count
