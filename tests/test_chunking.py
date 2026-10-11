from legal_rag_chatbot.ingestion.chunking import (
    MAX_WORDS,
    build_blocks,
    chunk_paragraphs,
    split_at_offsets,
    split_sentences,
)

SENTENCE = "Dies ist ein Satz mit genau neun Wörtern."


def text_of(sentences: int) -> str:
    return " ".join([SENTENCE] * sentences)


def test_build_blocks_merges_wrapped_text_and_drops_footers() -> None:
    paragraphs = [
        "1. Fristen: Die Kündigung ist am",
        "30.01.2026 zugegangen.",
        "a. Unterpunkt dazu.",
        "CASE-001-04  Seite 1",
        "2. Form: Schriftform gewahrt.",
    ]
    assert build_blocks(paragraphs) == [
        "1. Fristen: Die Kündigung ist am 30.01.2026 zugegangen. a. Unterpunkt dazu.",
        "2. Form: Schriftform gewahrt.",
    ]


def test_build_blocks_joins_a_paragraph_continued_after_a_page_break() -> None:
    paragraphs = [
        "Der Mandant hat die Maschine gekauft und",
        "CASE-003-01 Seite 1",
        "nutzt sie seit Dezember.",
    ]
    assert build_blocks(paragraphs) == [
        "Der Mandant hat die Maschine gekauft und nutzt sie seit Dezember."
    ]


def test_each_numbered_point_becomes_its_own_chunk() -> None:
    paragraphs = ["AKTENVERMERK | CASE-001 | Titel"] + [
        f"{n}. Thema {n}: {text_of(8)}" for n in range(1, 4)
    ]
    chunks = chunk_paragraphs(paragraphs)
    assert [c["section"] for c in chunks] == [
        "Punkt 1: Thema 1",
        "Punkt 2: Thema 2",
        "Punkt 3: Thema 3",
    ]
    assert chunks[0]["content"].startswith(
        "AKTENVERMERK | CASE-001 | Titel\n1. Thema 1:"
    )


def test_contract_clauses_are_labelled_with_number_and_title() -> None:
    paragraphs = [f"§ {n} Titel{n}: {text_of(8)}" for n in (1, 2)]
    sections = [c["section"] for c in chunk_paragraphs(paragraphs)]
    assert sections == ["§ 1 Titel1", "§ 2 Titel2"]


def test_small_points_are_merged_into_one_chunk() -> None:
    paragraphs = [f"{n}. Kurz {n}: {SENTENCE}" for n in range(1, 4)]
    chunks = chunk_paragraphs(paragraphs)
    assert len(chunks) == 1
    assert chunks[0]["section"] == "Punkt 1: Kurz 1 ff."


def test_oversized_point_is_split_at_sentence_boundaries() -> None:
    chunks = chunk_paragraphs([f"1. Lang: {text_of(70)}"])
    assert [c["section"] for c in chunks] == ["Punkt 1: Lang", "Punkt 1: Lang (Teil 2)"]
    assert all(len(c["content"].split()) <= MAX_WORDS for c in chunks)
    assert all(c["content"].endswith("Wörtern.") for c in chunks)


def test_split_at_offsets_cuts_the_text_into_paragraphs() -> None:
    assert split_at_offsets("abc\ndef\nghi", [4, 4, 8]) == ["abc\n", "def\n", "ghi"]


def test_footer_split_into_id_and_page_number_is_dropped() -> None:
    paragraphs = [
        "Ende des Absatzes.\n",
        "CASE-005-02\n",
        "Seite 1\n",
        "Neuer Absatz.\n",
    ]
    assert build_blocks(paragraphs) == ["Ende des Absatzes.", "Neuer Absatz."]


def test_clauses_in_one_ocr_paragraph_become_separate_blocks() -> None:
    paragraph = "§ 12 Kündigung: nach § 543 BGB.\n§ 13 Pflichten: Die Parteien\ninformieren sich."
    assert build_blocks([paragraph]) == [
        "§ 12 Kündigung: nach § 543 BGB.",
        "§ 13 Pflichten: Die Parteien informieren sich.",
    ]


def test_split_sentences_keeps_abbreviations_together() -> None:
    text = "Dr. Hollmann prüft Abs. 2 und z. B. Nr. 3 am 3. Mai. Danach folgt der nächste Satz."
    assert split_sentences(text) == [
        "Dr. Hollmann prüft Abs. 2 und z. B. Nr. 3 am 3. Mai.",
        "Danach folgt der nächste Satz.",
    ]
