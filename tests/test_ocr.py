from google.cloud import documentai

from legal_rag_chatbot.ingestion.ocr import to_record


def test_to_record_splits_text_per_page() -> None:
    text = "Seite eins\nSeite zwei\n"
    Segment = documentai.Document.TextAnchor.TextSegment

    def page(number: int, start: int, end: int, confidence: float):
        return documentai.Document.Page(
            page_number=number,
            layout=documentai.Document.Page.Layout(
                text_anchor=documentai.Document.TextAnchor(
                    text_segments=[Segment(start_index=start, end_index=end)]
                ),
                confidence=confidence,
            ),
        )

    document = documentai.Document(
        text=text, pages=[page(1, 0, 11, 0.98), page(2, 11, 22, 0.9)]
    )

    record = to_record("CASE-001-01.pdf", "projects/p/processors/x", document)

    assert record["document_id"] == "CASE-001-01"
    assert record["text"] == "Seite eins\nSeite zwei"
    assert [p["text"] for p in record["pages"]] == ["Seite eins", "Seite zwei"]
    assert record["pages"][1]["confidence"] == 0.9
