import json
from pathlib import Path

from legal_rag_chatbot.ingestion.pdf import build_pdfs, to_winansi


def test_to_winansi_replaces_characters_missing_in_standard_fonts() -> None:
    assert to_winansi("≈ 5 − 3 „ä“ €") == "ca. 5 – 3 „ä“ €"


def test_build_pdfs_writes_one_pdf_per_document(tmp_path: Path) -> None:
    doc = {
        "document_id": "CASE-001-01",
        "case_id": "CASE-001",
        "document_type": "interne_notiz",
        "document_title": "Aktenvermerk",
        "content": "AKTENVERMERK | CASE-001 | Titel\n1. Fristen: A & B <c>.\n\n§ 3 Miete: x.",
    }
    source = tmp_path / "docs.jsonl"
    source.write_text(json.dumps(doc) + "\n\n", encoding="utf-8")

    assert build_pdfs(source, tmp_path / "pdf") == 1
    assert (tmp_path / "pdf" / "CASE-001-01.pdf").read_bytes().startswith(b"%PDF")
