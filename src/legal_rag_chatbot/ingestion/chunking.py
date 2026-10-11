"""Split an OCR'd document into chunks along its structure (numbered points, clauses, paragraphs)."""

import math
import re

MAX_WORDS = 300
MIN_WORDS = 60

# Running page footers such as "CASE-001-04  Seite 2" are not content; Document AI often
# returns the id and the page number as separate paragraphs.
FOOTER = re.compile(r"^[A-Z]+-\d+(?:-\d+)?\s+Seite\s+\d+$")
PAGE_NUMBER = re.compile(r"^Seite\s+\d+$")
DOCUMENT_ID = re.compile(r"^[A-Z]+-\d+(?:-\d+)?$")
# A new numbered point ("3. ...") or contract clause ("§ 9 ...") always starts a block.
ITEM_START = re.compile(r"^(?:\d{1,2}\.|§\s*\d+[a-z]?)\s")
# Sub-points ("a. ...") and bullets belong to the point above them.
SUB_ITEM = re.compile(r"^(?:[a-z]\.|-)\s")
SENTENCE_END = (".", ":", ";", "!", "?", '"', "“", "”", ")")
ABBREVIATIONS = {
    "Dr.",
    "Prof.",
    "Nr.",
    "Abs.",
    "Art.",
    "ca.",
    "vgl.",
    "ggf.",
    "bzw.",
    "inkl.",
    "zzgl.",
    "gem.",
    "Ziff.",
    "lit.",
    "Rn.",
    "Az.",
    "Urt.",
    "Beschl.",
    "Mio.",
    "Tsd.",
}

# Ordinals ("am 3. Mai") and initials ("z. B.") do not end a sentence.
ORDINAL_OR_INITIAL = re.compile(r"(?:\d+|[A-Za-zÄÖÜäöü])\.")
CLAUSE_LABEL = re.compile(r"^§\s*(\d+[a-z]?)\s+([^:.]{1,60}):")
POINT_LABEL = re.compile(r"^(\d{1,2})\.\s+([^:.]{1,60}):")
CLAUSE_NUMBER = re.compile(r"^§\s*(\d+[a-z]?)\s")
POINT_NUMBER = re.compile(r"^(\d{1,2})\.\s")


def split_at_offsets(text: str, starts: list[int]) -> list[str]:
    """Cut the OCR text at the paragraph start offsets reported by the OCR engine."""
    bounds = sorted({0, len(text), *(s for s in starts if 0 <= s <= len(text))})
    return [text[a:b] for a, b in zip(bounds, bounds[1:])]


def _is_footer(texts: list[str], index: int) -> bool:
    following = texts[index + 1] if index + 1 < len(texts) else ""
    text = texts[index]
    return bool(
        FOOTER.match(text)
        or PAGE_NUMBER.match(text)
        or (DOCUMENT_ID.match(text) and PAGE_NUMBER.match(following))
    )


def build_blocks(paragraphs: list[str]) -> list[str]:
    """Join OCR paragraphs into logical blocks: one per point or clause, else per paragraph."""
    collapsed = [" ".join(p.split()) for p in paragraphs]
    blocks: list[str] = []
    for number, paragraph in enumerate(paragraphs):
        if not collapsed[number] or _is_footer(collapsed, number):
            continue
        # OCR often puts consecutive points into one paragraph, so check every line.
        lines = [" ".join(line.split()) for line in paragraph.split("\n")]
        for index, line in enumerate(line for line in lines if line):
            starts_item = ITEM_START.match(line) and (
                index == 0 or blocks[-1].endswith(SENTENCE_END)
            )
            if starts_item or not blocks:
                blocks.append(line)
            elif index > 0 or _continues(blocks[-1], line):
                blocks[-1] += " " + line
            else:
                blocks.append(line)
    return blocks


def _continues(previous: str, text: str) -> bool:
    """True if the paragraph continues the previous one (sub-point, wrapped sentence, page break)."""
    return (
        bool(SUB_ITEM.match(text))
        or text[0].islower()
        or not previous.endswith(SENTENCE_END)
    )


def _label(block: str) -> str | None:
    if m := CLAUSE_LABEL.match(block):
        return f"§ {m[1]} {m[2].strip()}"
    if m := POINT_LABEL.match(block):
        return f"Punkt {m[1]}: {m[2].strip()}"
    if m := CLAUSE_NUMBER.match(block):
        return f"§ {m[1]}"
    if m := POINT_NUMBER.match(block):
        return f"Punkt {m[1]}"
    return None


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    words = text.split()
    for index, word in enumerate(words):
        current.append(word)
        following = words[index + 1] if index + 1 < len(words) else ""
        ends_sentence = (
            word.endswith((".", "!", "?"))
            and word not in ABBREVIATIONS
            and not ORDINAL_OR_INITIAL.fullmatch(word)
            and following[:1].isupper()
        )
        if ends_sentence:
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return sentences


def split_oversized(text: str) -> list[str]:
    """Split a block above MAX_WORDS into pieces of similar size at sentence boundaries."""
    total = len(text.split())
    if total <= MAX_WORDS:
        return [text]
    target = total / math.ceil(total / MAX_WORDS)
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in split_sentences(text):
        current.append(sentence)
        count += len(sentence.split())
        if count >= target:
            pieces.append(" ".join(current))
            current, count = [], 0
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_paragraphs(paragraphs: list[str]) -> list[dict[str, str]]:
    """Return chunks as {"section", "content"} dicts of at most about MAX_WORDS words."""
    chunks: list[dict] = []
    current: dict | None = None
    for block in build_blocks(paragraphs):
        label = _label(block)
        for index, piece in enumerate(split_oversized(block)):
            piece_label = (
                label if index == 0 else f"{label or 'Absatz'} (Teil {index + 1})"
            )
            words = len(piece.split())
            starts_item = index == 0 and label is not None
            if current and (
                current["words"] + words > MAX_WORDS
                or (starts_item and current["words"] >= MIN_WORDS)
            ):
                chunks.append(current)
                current = None
            if current is None:
                current = {"labels": [], "parts": [], "words": 0}
            if piece_label:
                current["labels"].append(piece_label)
            current["parts"].append(piece)
            current["words"] += words
    if current:
        chunks.append(current)

    if (
        len(chunks) > 1
        and chunks[-1]["words"] < MIN_WORDS
        and chunks[-2]["words"] + chunks[-1]["words"] <= MAX_WORDS
    ):
        last = chunks.pop()
        for key in ("labels", "parts"):
            chunks[-1][key] += last[key]
        chunks[-1]["words"] += last["words"]

    return [
        {
            "section": _section(chunk["labels"], number),
            "content": "\n".join(chunk["parts"]),
        }
        for number, chunk in enumerate(chunks, start=1)
    ]


def _section(labels: list[str], number: int) -> str:
    if not labels:
        return f"Abschnitt {number}"
    return labels[0] if len(labels) == 1 else f"{labels[0]} ff."
