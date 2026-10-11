"""Render the dataset documents as PDFs so they look like real case-file documents."""

import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

TITLE = ParagraphStyle(
    "title", fontName=FONT_BOLD, fontSize=14, leading=18, spaceAfter=3
)
SUBTITLE = ParagraphStyle(
    "subtitle",
    fontName=FONT,
    fontSize=8.5,
    leading=11,
    textColor=colors.grey,
    spaceAfter=14,
)
BODY = ParagraphStyle("body", fontName=FONT, fontSize=10, leading=14, spaceAfter=3)

# The standard PDF fonts only cover Windows-1252, so these two characters need a stand-in.
REPLACEMENTS = {"≈": "ca.", "−": "–"}
HANGING = re.compile(r"^(?:\d{1,2}\.|[a-z]\.|-)\s")
CLAUSE_LABEL = re.compile(r"^(§ \d+[a-z]? [^:]{1,60}:)")
LABEL = re.compile(r"^((?:\d{1,2}\.|[a-z]\.)\s+)?([^:.\d]{2,60}:)")


def to_winansi(text: str) -> str:
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _markup(line: str) -> str:
    """Escape the line for Paragraph and bold a leading label such as '§ 3 Miete:'."""
    text = escape(to_winansi(line))
    if match := CLAUSE_LABEL.match(text):
        return f"<b>{match.group(1)}</b>{text[match.end() :]}"
    if match := LABEL.match(text):
        number = match.group(1) or ""
        return f"{number}<b>{match.group(2)}</b>{text[match.end() :]}"
    return text


def body_flowables(body: str) -> list:
    flowables = []
    for line in body.split("\n"):
        if not line.strip():
            flowables.append(Spacer(1, 5))
            continue
        stripped = line.strip()
        indent = (len(line) - len(line.lstrip(" "))) * 3
        if HANGING.match(stripped):
            style = ParagraphStyle(
                "item", parent=BODY, leftIndent=indent + 16, firstLineIndent=-16
            )
        else:
            style = ParagraphStyle("line", parent=BODY, leftIndent=indent)
        flowables.append(Paragraph(_markup(stripped), style))
    return flowables


def render_pdf(doc: dict, path: Path) -> None:
    # The first content line is a header like "AKTENVERMERK | CASE-001 | Titel | Datum".
    first, _, body = doc["content"].partition("\n")
    heading, *details = first.split(" | ")
    story = [Paragraph(escape(to_winansi(heading)), TITLE)]
    if details:
        story.append(Paragraph(escape(to_winansi("  |  ".join(details))), SUBTITLE))
    story += body_flowables(body.lstrip("\n"))

    def footer(canvas, pdf_doc) -> None:
        canvas.setFont(FONT, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2.2 * cm, 1.3 * cm, doc["document_id"])
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, f"Seite {pdf_doc.page}")

    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title=doc["document_title"],
        subject=f"{doc['document_type']} {doc['case_id'] or ''}".strip(),
        keywords="synthetic, fictional",
        invariant=True,
    ).build(story, onFirstPage=footer, onLaterPages=footer)


def build_pdfs(source: str | Path, destination: str | Path) -> int:
    """Write one PDF named after the document id per JSONL document and return the count."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    count = 0
    with Path(source).open(encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            doc = json.loads(line)
            render_pdf(doc, destination / f"{doc['document_id']}.pdf")
            count += 1
    return count
