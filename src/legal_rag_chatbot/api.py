from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from legal_rag_chatbot.evaluation.dataset import load_corpus
from legal_rag_chatbot.ingestion.extract import ExtractionError
from legal_rag_chatbot.llm.chat import answer_with_sources
from legal_rag_chatbot.rag import session

EXCERPT_CHARS = 300
PDF_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "pdf"

app = FastAPI(title="legal-rag-chatbot")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    document_ids: list[str] = Field(default_factory=list)


class SourceOut(BaseModel):
    number: int
    documentId: str
    documentName: str
    excerpt: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class UploadOut(BaseModel):
    id: str
    name: str
    type: str
    size: int
    uploaded_at: str


@lru_cache(maxsize=1)
def document_titles() -> dict[str, str]:
    return {doc_id: doc["document_title"] for doc_id, doc in load_corpus().items()}


@app.post("/api/documents/upload")
def upload_document(file: UploadFile) -> UploadOut:
    """Store an upload for this session only; it is not added to the RAG corpus."""
    data = file.file.read(session.MAX_UPLOAD_BYTES + 1)
    if len(data) > session.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Die Datei ist größer als 20 MB.")
    try:
        document = session.store.add(file.filename or "", data)
    except ExtractionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return UploadOut(
        id=document.id,
        name=document.name,
        type=document.media_type,
        size=len(document.content),
        uploaded_at=document.uploaded_at,
    )


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    session.store.remove(document_id)


@app.get("/api/documents/{document_id}/file")
def document_file(document_id: str) -> Response:
    if upload := session.store.get(document_id):
        return Response(
            upload.content,
            media_type=upload.media_type,
            headers={
                "Content-Disposition": f"inline; filename*=UTF-8''{quote(upload.name)}",
                "X-Content-Type-Options": "nosniff",
            },
        )
    # Only IDs from the corpus are served, which rules out path traversal.
    if document_id not in document_titles():
        raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
    path = PDF_DIR / f"{document_id}.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="PDF nicht vorhanden")
    return FileResponse(path, media_type="application/pdf")


@app.post("/api/chat", response_model_exclude_none=True)
def chat(request: ChatRequest) -> ChatResponse:
    answer = answer_with_sources(
        request.question, uploaded_document_ids=request.document_ids
    )
    titles = document_titles()
    sources = [
        SourceOut(
            number=source.number,
            documentId=source.document_id or "",
            documentName=(
                source.document_name
                or (
                    f"{source.document_id} – {titles[source.document_id]}"
                    if source.document_id in titles
                    else "Unbekanntes Dokument"
                )
            ),
            excerpt=source.text[:EXCERPT_CHARS],
            page=source.page,
        )
        for source in answer.cited_sources()
    ]
    return ChatResponse(answer=answer.text, sources=sources)
