import json
from pathlib import Path

# Fields that end up in the embedded chunk text; ids and flags would only dilute it.
EMBEDDED_FIELDS = (
    "document_id",
    "case_id",
    "document_type",
    "document_title",
    "created_at",
    "content",
)


def load_documents(source: str | Path, ocr_dir: str | Path) -> list[dict]:
    """Read the raw documents and replace their content with the OCR text of their PDF."""
    with Path(source).open(encoding="utf-8") as src:
        docs = [json.loads(line) for line in src if line.strip()]
    for doc in docs:
        ocr_file = Path(ocr_dir) / f"{doc['document_id']}.json"
        if not ocr_file.is_file():
            raise FileNotFoundError(
                f"No OCR result for {doc['document_id']} at {ocr_file}; run the OCR script first"
            )
        doc["content"] = json.loads(ocr_file.read_text(encoding="utf-8"))["text"]
    return docs


def build_embedding_file(
    source: str | Path, ocr_dir: str | Path, destination: str | Path
) -> int:
    """Write a JSONL with only the embedded fields per document and return the count."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    docs = load_documents(source, ocr_dir)
    with destination.open("w", encoding="utf-8") as dst:
        for doc in docs:
            slim = {field: doc[field] for field in EMBEDDED_FIELDS}
            dst.write(json.dumps(slim, ensure_ascii=False) + "\n")
    return len(docs)
