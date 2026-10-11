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


def _write_ocr(ocr_dir: Path, **fields) -> None:
    ocr_dir.mkdir()
    (ocr_dir / "document_id-value.json").write_text(
        json.dumps({"document_id": "document_id-value", **fields}),
        encoding="utf-8",
    )


def test_build_embedding_file_writes_one_record_per_chunk(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    ocr_dir = tmp_path / "ocr"
    text = "Erster Absatz.\nCASE-001-01\nSeite 1"
    _write_ocr(ocr_dir, text=text, paragraph_starts=[0, 15, 27])

    destination = tmp_path / "out" / "chunks.jsonl"
    assert build_embedding_file(source, ocr_dir, destination) == 1

    written = json.loads(destination.read_text(encoding="utf-8"))
    assert tuple(written) == EMBEDDED_FIELDS
    assert written["content"] == "Erster Absatz."
    assert written["section"] == "Abschnitt 1"
    assert written["case_id"] == "case_id-value"


def test_build_embedding_file_requires_paragraph_offsets_in_the_ocr_result(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    ocr_dir = tmp_path / "ocr"
    _write_ocr(ocr_dir, text="no offsets here")
    with pytest.raises(ValueError, match="--force"):
        build_embedding_file(source, ocr_dir, tmp_path / "chunks.jsonl")


def test_build_embedding_file_requires_an_ocr_result(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    with pytest.raises(FileNotFoundError, match="OCR"):
        build_embedding_file(source, tmp_path / "ocr", tmp_path / "slim.jsonl")
