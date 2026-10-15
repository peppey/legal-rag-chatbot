import json
from pathlib import Path

import pytest

from legal_rag_chatbot.access import GROUPS, chunk_file_name
from legal_rag_chatbot.ingestion.prepare import EMBEDDED_FIELDS, build_embedding_files


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


def test_build_embedding_files_writes_the_chunks_into_the_file_of_their_group(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    ocr_dir = tmp_path / "ocr"
    text = "Erster Absatz.\nCASE-001-01\nSeite 1"
    _write_ocr(ocr_dir, text=text, paragraph_starts=[0, 15, 27], access_group="B")

    out = tmp_path / "out"
    assert build_embedding_files(source, ocr_dir, out) == {"A": 0, "B": 1, "Geteilt": 0}

    assert (out / chunk_file_name("A")).read_text(encoding="utf-8") == ""
    written = json.loads((out / chunk_file_name("B")).read_text(encoding="utf-8"))
    assert tuple(written) == EMBEDDED_FIELDS
    assert written["content"] == "Erster Absatz."
    assert written["section"] == "Abschnitt 1"
    assert written["case_id"] == "case_id-value"
    assert all((out / chunk_file_name(g)).is_file() for g in GROUPS)


def test_build_embedding_files_requires_an_access_group_in_the_ocr_result(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    ocr_dir = tmp_path / "ocr"
    _write_ocr(ocr_dir, text="Absatz", paragraph_starts=[0])
    with pytest.raises(ValueError, match="access group"):
        build_embedding_files(source, ocr_dir, tmp_path / "out")


def test_build_embedding_files_requires_paragraph_offsets_in_the_ocr_result(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    ocr_dir = tmp_path / "ocr"
    _write_ocr(ocr_dir, text="no offsets here", access_group="A")
    with pytest.raises(ValueError, match="--force"):
        build_embedding_files(source, ocr_dir, tmp_path / "out")


def test_build_embedding_files_requires_an_ocr_result(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    with pytest.raises(FileNotFoundError, match="OCR"):
        build_embedding_files(source, tmp_path / "ocr", tmp_path / "out")
