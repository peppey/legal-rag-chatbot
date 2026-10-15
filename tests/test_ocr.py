from pathlib import Path

import pytest
from google.cloud import documentai

from legal_rag_chatbot.ingestion.ocr import find_pdfs, to_record


def test_to_record_splits_text_per_page_and_records_paragraph_starts() -> None:
    text = "Seite eins\nSeite zwei\n"
    Segment = documentai.Document.TextAnchor.TextSegment

    def layout(start: int, end: int, confidence: float = 0.0):
        return documentai.Document.Page.Layout(
            text_anchor=documentai.Document.TextAnchor(
                text_segments=[Segment(start_index=start, end_index=end)]
            ),
            confidence=confidence,
        )

    def page(number: int, start: int, end: int, confidence: float):
        return documentai.Document.Page(
            page_number=number,
            layout=layout(start, end, confidence),
            paragraphs=[documentai.Document.Page.Paragraph(layout=layout(start, end))],
        )

    document = documentai.Document(
        text=text, pages=[page(1, 0, 11, 0.98), page(2, 11, 22, 0.9)]
    )

    record = to_record("CASE-001-01.pdf", "projects/p/processors/x", document, "A")

    assert record["document_id"] == "CASE-001-01"
    assert record["access_group"] == "A"
    assert record["text"] == "Seite eins\nSeite zwei"
    assert [p["text"] for p in record["pages"]] == ["Seite eins", "Seite zwei"]
    assert record["pages"][1]["confidence"] == 0.9
    assert record["paragraph_starts"] == [0, 11]


def _touch(pdf_dir: Path, *paths: str) -> None:
    for path in paths:
        (pdf_dir / path).parent.mkdir(parents=True, exist_ok=True)
        (pdf_dir / path).write_bytes(b"%PDF")


def test_find_pdfs_reads_the_group_folders(tmp_path: Path) -> None:
    _touch(
        tmp_path, "A/CASE-001-01.pdf", "B/CASE-002-01.pdf", "Geteilt/CASE-003-01.pdf"
    )

    assert [p.relative_to(tmp_path).as_posix() for p in find_pdfs(tmp_path)] == [
        "A/CASE-001-01.pdf",
        "B/CASE-002-01.pdf",
        "Geteilt/CASE-003-01.pdf",
    ]


def test_find_pdfs_rejects_unknown_group_folders(tmp_path: Path) -> None:
    _touch(tmp_path, "A/CASE-001-01.pdf", "C/CASE-002-01.pdf")
    with pytest.raises(ValueError, match="Unknown group"):
        find_pdfs(tmp_path)


def test_find_pdfs_rejects_a_case_in_two_groups(tmp_path: Path) -> None:
    _touch(tmp_path, "A/CASE-001-01.pdf", "B/CASE-001-02.pdf")
    with pytest.raises(ValueError, match="CASE-001"):
        find_pdfs(tmp_path)
