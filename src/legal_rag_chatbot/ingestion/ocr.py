"""OCR the exported PDFs with Google Cloud Document AI and store the text as JSON."""

import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.api_core import exceptions, retry
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from pypdf import PdfReader, PdfWriter

from legal_rag_chatbot.access import GROUPS
from legal_rag_chatbot.ingestion.chunking import split_at_offsets
from legal_rag_chatbot.ingestion.extract import ExtractionError, TextUnit
from legal_rag_chatbot.rag.query import PROJECT_ID

logger = logging.getLogger(__name__)

PROCESSOR_TYPE = "OCR_PROCESSOR"
# Document AI is not available in europe-west3; "eu" keeps the data in the EU.
OCR_LOCATION = "eu"
OCR_PROCESSOR_NAME = "legal-rag-ocr"
LANGUAGE_HINTS = ["de"]
# Online processing accepts 15 pages per request; longer uploads are sent in slices.
PAGES_PER_REQUEST = 15
MAX_UPLOAD_PAGES = 60
RETRY = retry.Retry(
    predicate=retry.if_exception_type(
        exceptions.ResourceExhausted,
        exceptions.ServiceUnavailable,
        exceptions.InternalServerError,
    ),
    initial=2.0,
    maximum=60.0,
    timeout=300.0,
)


def create_client(location: str) -> documentai.DocumentProcessorServiceClient:
    return documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com"
        )
    )


def get_or_create_processor(
    client: documentai.DocumentProcessorServiceClient,
    project_id: str,
    location: str,
    display_name: str,
) -> str:
    """Return the resource name of the OCR processor, creating it on first use."""
    parent = client.common_location_path(project_id, location)
    for processor in client.list_processors(parent=parent):
        if (
            processor.display_name == display_name
            and processor.type_ == PROCESSOR_TYPE
            and processor.state == documentai.Processor.State.ENABLED
        ):
            return processor.name
    created = client.create_processor(
        parent=parent,
        processor=documentai.Processor(display_name=display_name, type_=PROCESSOR_TYPE),
    )
    print(f"Created OCR processor {created.name}")
    return created.name


def ocr_pdf(
    client: documentai.DocumentProcessorServiceClient,
    processor_name: str,
    content: bytes,
) -> documentai.Document:
    """Run online OCR on one PDF (the API accepts up to 15 pages per request)."""
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(
            content=content, mime_type="application/pdf"
        ),
        process_options=documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                hints=documentai.OcrConfig.Hints(language_hints=LANGUAGE_HINTS)
            )
        ),
    )
    return client.process_document(request=request, retry=RETRY).document


def _anchor_text(text: str, anchor: documentai.Document.TextAnchor) -> str:
    return "".join(
        text[int(segment.start_index) : int(segment.end_index)]
        for segment in anchor.text_segments
    )


def find_pdfs(pdf_dir: Path) -> list[Path]:
    """PDFs in the group folders below pdf_dir; all documents of one case must share a group."""
    unknown = [d.name for d in pdf_dir.iterdir() if d.is_dir() and d.name not in GROUPS]
    if unknown:
        raise ValueError(f"Unknown group folders in {pdf_dir}: {sorted(unknown)}")
    pdfs = sorted(pdf for group in GROUPS for pdf in (pdf_dir / group).glob("*.pdf"))
    case_groups: dict[str, set[str]] = {}
    for pdf in pdfs:
        case_id = pdf.stem.rsplit("-", 1)[0]
        case_groups.setdefault(case_id, set()).add(pdf.parent.name)
    split = {case: sorted(g) for case, g in case_groups.items() if len(g) > 1}
    if split:
        raise ValueError(f"Cases spread over several groups: {split}")
    return pdfs


def to_record(
    pdf_name: str,
    processor_name: str,
    document: documentai.Document,
    access_group: str,
) -> dict:
    """Convert a Document AI result into the JSON record stored per PDF."""
    leading = len(document.text) - len(document.text.lstrip())
    # Paragraph boundaries as offsets into the stripped text. The per-page paragraph lists of
    # Document AI can overlap and repeat text, the plain text is the reliable source.
    paragraph_starts = sorted(
        {
            max(int(segments[0].start_index) - leading, 0)
            for page in document.pages
            for paragraph in page.paragraphs
            if (segments := paragraph.layout.text_anchor.text_segments)
        }
    )
    pages = [
        {
            "page_number": page.page_number,
            "text": _anchor_text(document.text, page.layout.text_anchor).strip(),
            "confidence": round(page.layout.confidence, 4),
        }
        for page in document.pages
    ]
    return {
        "document_id": Path(pdf_name).stem,
        "source_pdf": pdf_name,
        "access_group": access_group,
        "text": document.text.strip(),
        "paragraph_starts": paragraph_starts,
        "pages": pages,
        "ocr": {
            "engine": "google-cloud-document-ai",
            "processor": processor_name,
            "language_hints": LANGUAGE_HINTS,
        },
    }


def document_units(
    document: documentai.Document, first_page: int = 1
) -> list[TextUnit]:
    """Paragraphs per page of an OCR result, in the form the upload store expects."""
    units: list[TextUnit] = []
    for page in document.pages:
        number = first_page + page.page_number - 1
        segments = page.layout.text_anchor.text_segments
        if not segments:
            units.append((number, []))
            continue
        start = min(int(s.start_index) for s in segments)
        end = max(int(s.end_index) for s in segments)
        starts = [
            int(p.layout.text_anchor.text_segments[0].start_index) - start
            for p in page.paragraphs
            if p.layout.text_anchor.text_segments
        ]
        pieces = split_at_offsets(document.text[start:end], starts)
        units.append((number, [p.strip() for p in pieces if p.strip()]))
    return units


_upload_ocr: tuple[documentai.DocumentProcessorServiceClient, str] | None = None
_upload_ocr_lock = threading.Lock()


def _upload_processor() -> tuple[documentai.DocumentProcessorServiceClient, str]:
    global _upload_ocr
    with _upload_ocr_lock:
        if _upload_ocr is None:
            client = create_client(OCR_LOCATION)
            _upload_ocr = (
                client,
                get_or_create_processor(
                    client, PROJECT_ID, OCR_LOCATION, OCR_PROCESSOR_NAME
                ),
            )
        return _upload_ocr


def _slices(data: bytes, reader: PdfReader) -> list[bytes]:
    if len(reader.pages) <= PAGES_PER_REQUEST:
        return [data]
    slices = []
    for first in range(0, len(reader.pages), PAGES_PER_REQUEST):
        writer = PdfWriter()
        for page in reader.pages[first : first + PAGES_PER_REQUEST]:
            writer.add_page(page)
        buffer = io.BytesIO()
        writer.write(buffer)
        slices.append(buffer.getvalue())
    return slices


def ocr_units(data: bytes) -> list[TextUnit]:
    """OCR an uploaded PDF without a text layer. Raises ExtractionError if that fails."""
    reader = PdfReader(io.BytesIO(data))
    if len(reader.pages) > MAX_UPLOAD_PAGES:
        raise ExtractionError(
            f"Gescannte PDFs mit mehr als {MAX_UPLOAD_PAGES} Seiten werden nicht unterstützt."
        )
    units: list[TextUnit] = []
    try:
        client, processor_name = _upload_processor()
        for index, part in enumerate(_slices(data, reader)):
            document = ocr_pdf(client, processor_name, part)
            units += document_units(document, first_page=index * PAGES_PER_REQUEST + 1)
    except exceptions.GoogleAPICallError as error:
        logger.exception("OCR of an upload failed")
        raise ExtractionError(
            "Die Texterkennung für das gescannte PDF ist fehlgeschlagen."
        ) from error
    return units


def run_ocr(
    pdf_dir: Path,
    output_dir: Path,
    project_id: str,
    location: str,
    display_name: str,
    workers: int = 4,
    force: bool = False,
    limit: int | None = None,
) -> int:
    """OCR all PDFs without an existing result (or all with force) and return how many succeeded."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = [
        pdf
        for pdf in find_pdfs(pdf_dir)
        if force or not (output_dir / f"{pdf.stem}.json").exists()
    ][:limit]
    if not pdfs:
        print("Nothing to do, all PDFs already have an OCR result.")
        return 0

    client = create_client(location)
    processor_name = get_or_create_processor(client, project_id, location, display_name)

    def work(pdf: Path) -> None:
        record = to_record(
            pdf.name,
            processor_name,
            ocr_pdf(client, processor_name, pdf.read_bytes()),
            pdf.parent.name,
        )
        (output_dir / f"{pdf.stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, pdf): pdf for pdf in pdfs}
        for future in as_completed(futures):
            pdf = futures[future]
            try:
                future.result()
            except exceptions.GoogleAPICallError as exc:
                failed.append(pdf.name)
                print(f"FAILED {pdf.name}: {exc}")
                continue
            done += 1
            print(f"[{done}/{len(pdfs)}] {pdf.name}")
    if failed:
        print(f"{len(failed)} PDFs failed, run again to retry: {', '.join(failed)}")
    return done
