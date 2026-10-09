import json
from pathlib import Path

from legal_rag_chatbot.ingestion.prepare import EMBEDDED_FIELDS, build_embedding_file


def test_build_embedding_file_keeps_only_embedded_fields(tmp_path: Path) -> None:
    doc = {field: f"{field}-value" for field in EMBEDDED_FIELDS}
    doc.update({"client_id": "CLIENT-001", "related_document_ids": ["X"]})
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(doc) + "\n\n", encoding="utf-8")

    destination = tmp_path / "out" / "slim.jsonl"
    assert build_embedding_file(source, destination) == 1

    written = json.loads(destination.read_text(encoding="utf-8"))
    assert tuple(written) == EMBEDDED_FIELDS
