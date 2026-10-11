"""OCR the exported PDFs with Google Cloud Document AI and store the text as JSON."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.api_core import exceptions, retry
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

PROCESSOR_TYPE = "OCR_PROCESSOR"
LANGUAGE_HINTS = ["de"]
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
    pdf_path: Path,
) -> documentai.Document:
    """Run online OCR on one PDF (the API accepts up to 15 pages per request)."""
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(
            content=pdf_path.read_bytes(), mime_type="application/pdf"
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


def to_record(
    pdf_name: str, processor_name: str, document: documentai.Document
) -> dict:
    """Convert a Document AI result into the JSON record stored per PDF."""
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
        "text": document.text.strip(),
        "pages": pages,
        "ocr": {
            "engine": "google-cloud-document-ai",
            "processor": processor_name,
            "language_hints": LANGUAGE_HINTS,
        },
    }


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
        for pdf in sorted(pdf_dir.glob("*.pdf"))
        if force or not (output_dir / f"{pdf.stem}.json").exists()
    ][:limit]
    if not pdfs:
        print("Nothing to do, all PDFs already have an OCR result.")
        return 0

    client = create_client(location)
    processor_name = get_or_create_processor(client, project_id, location, display_name)

    def work(pdf: Path) -> None:
        record = to_record(
            pdf.name, processor_name, ocr_pdf(client, processor_name, pdf)
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
