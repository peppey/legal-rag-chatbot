import io

import pytest
from google.api_core import exceptions
from google.cloud import documentai
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from legal_rag_chatbot.ingestion import ocr
from legal_rag_chatbot.ingestion.extract import ExtractionError, extract_text_units


def _text_pdf(text: str) -> bytes:
    buffer = io.BytesIO()
    page = canvas.Canvas(buffer)
    page.drawString(72, 720, text)
    page.save()
    return buffer.getvalue()


def _blank_pdf(pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _fail(data: bytes):
    raise AssertionError("OCR must not be called")


def test_a_pdf_with_text_layer_is_not_sent_to_ocr() -> None:
    units = extract_text_units(
        "miete.pdf", _text_pdf("Die Miete betraegt 900 Euro."), _fail
    )

    assert [(page, [p.strip() for p in paragraphs]) for page, paragraphs in units] == [
        (1, ["Die Miete betraegt 900 Euro."])
    ]


def test_a_pdf_without_text_layer_falls_back_to_ocr() -> None:
    units = extract_text_units(
        "scan.pdf", _blank_pdf(), lambda data: [(1, ["Gescannter Text"]), (2, [])]
    )

    assert units == [(1, ["Gescannter Text"])]


@pytest.mark.parametrize("fallback", [None, lambda data: [(1, [])]])
def test_a_pdf_without_any_text_is_rejected(fallback) -> None:
    with pytest.raises(ExtractionError, match="kein Text"):
        extract_text_units("scan.pdf", _blank_pdf(), fallback)


def test_other_formats_never_use_ocr() -> None:
    with pytest.raises(ExtractionError, match="kein Text"):
        extract_text_units("leer.txt", b"  \n", _fail)


def _document() -> documentai.Document:
    text = "Seite eins\nAbsatz A\nSeite zwei\n"
    Segment = documentai.Document.TextAnchor.TextSegment

    def layout(*spans: tuple[int, int]):
        segments = [Segment(start_index=a, end_index=b) for a, b in spans]
        return documentai.Document.Page.Layout(
            text_anchor=documentai.Document.TextAnchor(text_segments=segments)
        )

    def paragraph(start: int, end: int):
        return documentai.Document.Page.Paragraph(layout=layout((start, end)))

    return documentai.Document(
        text=text,
        pages=[
            documentai.Document.Page(
                page_number=1,
                layout=layout((0, 20)),
                paragraphs=[paragraph(0, 11), paragraph(11, 20)],
            ),
            documentai.Document.Page(
                page_number=2,
                layout=layout((20, 31)),
                paragraphs=[paragraph(20, 31)],
            ),
        ],
    )


def test_document_units_split_the_pages_into_paragraphs() -> None:
    assert ocr.document_units(_document()) == [
        (1, ["Seite eins", "Absatz A"]),
        (2, ["Seite zwei"]),
    ]
    assert [page for page, _ in ocr.document_units(_document(), first_page=16)] == [
        16,
        17,
    ]


def test_long_scans_are_sent_in_slices_with_continuing_page_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_pages: list[int] = []

    def fake_ocr_pdf(client, processor, content: bytes):
        count = len(PdfReader(io.BytesIO(content)).pages)
        sent_pages.append(count)
        pages = [documentai.Document.Page(page_number=n) for n in range(1, count + 1)]
        return documentai.Document(pages=pages)

    monkeypatch.setattr(ocr, "_upload_processor", lambda: (None, "processor"))
    monkeypatch.setattr(ocr, "ocr_pdf", fake_ocr_pdf)

    units = ocr.ocr_units(_blank_pdf(20))

    assert sent_pages == [15, 5]
    assert [page for page, _ in units] == list(range(1, 21))


def test_ocr_rejects_too_long_scans_and_reports_api_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ocr, "_upload_processor", lambda: (None, "processor"))

    def unavailable(client, processor, content):
        raise exceptions.ServiceUnavailable("down")

    monkeypatch.setattr(ocr, "ocr_pdf", unavailable)

    with pytest.raises(ExtractionError, match="Texterkennung"):
        ocr.ocr_units(_blank_pdf())
    with pytest.raises(ExtractionError, match="mehr als"):
        ocr.ocr_units(_blank_pdf(ocr.MAX_UPLOAD_PAGES + 1))
