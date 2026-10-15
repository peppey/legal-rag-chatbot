import json
from pathlib import Path

from legal_rag_chatbot.access import GROUPS, chunk_file_name
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


def build_embedding_files(
    source: str | Path, ocr_dir: str | Path, destination_dir: str | Path
) -> dict[str, int]:
    """Write one chunk file per access group and return the chunk count per group.

    The group is not part of the embedded text: it is the file a chunk lives in, and the
    retrieval restricts the search to the files of the groups a user may read.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(GROUPS, 0)
    files = {
        group: (destination_dir / chunk_file_name(group)).open("w", encoding="utf-8")
        for group in GROUPS
    }
    try:
        with Path(source).open(encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                doc = json.loads(line)
                ocr = load_ocr(ocr_dir, doc["document_id"])
                group = ocr.get("access_group")
                if group not in GROUPS:
                    raise ValueError(
                        f"OCR result of {doc['document_id']} has no valid access group "
                        f"({group!r}); rerun the OCR with --force"
                    )
                chunks = chunk_paragraphs(_paragraphs(ocr))
                if not chunks:
                    raise ValueError(
                        f"No text found in the OCR result of {doc['document_id']}"
                    )
                for chunk in chunks:
                    record = {**doc, **chunk}
                    slim = {field: record[field] for field in EMBEDDED_FIELDS}
                    files[group].write(json.dumps(slim, ensure_ascii=False) + "\n")
                    counts[group] += 1
    finally:
        for file in files.values():
            file.close()
    return counts
