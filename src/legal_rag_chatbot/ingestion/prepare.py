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


def build_embedding_file(source: str | Path, destination: str | Path) -> int:
    """Write a JSONL with only the embedded fields per document and return the count."""
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
            slim = {field: doc[field] for field in EMBEDDED_FIELDS}
            dst.write(json.dumps(slim, ensure_ascii=False) + "\n")
            count += 1
    return count
