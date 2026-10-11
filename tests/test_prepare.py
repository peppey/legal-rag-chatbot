import json
from pathlib import Path

import pytest

from legal_rag_chatbot.ingestion.prepare import EMBEDDED_FIELDS, build_embedding_file


def _write_source(tmp_path: Path) -> Path:
    doc = {field: f"{field}-value" for field in EMBEDDED_FIELDS}
    doc.update({"client_id": "CLIENT-001", "related_document_ids": ["X"]})
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(doc) + "\n\n", encoding="utf-8")
    return source


def test_build_embedding_file_uses_ocr_text_and_keeps_only_embedded_fields(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "document_id-value.json").write_text(
        json.dumps({"text": "ocr text"}), encoding="utf-8"
    )

    destination = tmp_path / "out" / "slim.jsonl"
    assert build_embedding_file(source, ocr_dir, destination) == 1

    written = json.loads(destination.read_text(encoding="utf-8"))
    assert tuple(written) == EMBEDDED_FIELDS
    assert written["content"] == "ocr text"
    assert written["case_id"] == "case_id-value"


def test_build_embedding_file_requires_an_ocr_result(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    with pytest.raises(FileNotFoundError, match="OCR"):
        build_embedding_file(source, tmp_path / "ocr", tmp_path / "slim.jsonl")
