import json
import logging
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from legal_rag_chatbot.access import PDF_DIR, Access, document_groups
from legal_rag_chatbot.auth import current_access, warn_if_dev_mode
from legal_rag_chatbot.evaluation.dataset import load_corpus
from legal_rag_chatbot.ingestion.extract import ExtractionError
from legal_rag_chatbot.llm.chat import (
    Answer,
    SearchMode,
    Source,
    Turn,
    answer_with_sources,
    prepare_answer,
    stream_answer,
)
from legal_rag_chatbot.rag import session

logger = logging.getLogger(__name__)

EXCERPT_CHARS = 300
BUSY_MESSAGE = (
    "Der Dienst ist gerade ausgelastet. Bitte versuche es in einer Minute erneut."
)
FAILED_MESSAGE = "Die Antwort konnte nicht erstellt werden."


def _is_busy(error: Exception) -> bool:
    """Quota exhausted (429) or service unavailable (503), also after the client retries."""
    return isinstance(error, genai_errors.APIError) and error.code in (429, 503)


app = FastAPI(title="legal-rag-chatbot")
warn_if_dev_mode()


@app.exception_handler(genai_errors.APIError)
def gemini_error(request: Request, error: genai_errors.APIError) -> JSONResponse:
    """Turn Gemini errors before the answer starts into a message the frontend can show."""
    logger.warning("Gemini request failed: %s", error)
    if _is_busy(error):
        return JSONResponse({"detail": BUSY_MESSAGE}, status_code=503)
    return JSONResponse({"detail": FAILED_MESSAGE}, status_code=502)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=50_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    document_ids: list[str] = Field(default_factory=list)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)
    search_mode: SearchMode = "auto"

    def turns(self) -> list[Turn]:
        return [Turn(m.role, m.content) for m in self.history]


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
def upload_document(
    file: UploadFile, access: Access = Depends(current_access)
) -> UploadOut:
    """Store an upload for this user's session only; it is not added to the RAG corpus."""
    data = file.file.read(session.MAX_UPLOAD_BYTES + 1)
    if len(data) > session.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Die Datei ist größer als 20 MB.")
    try:
        document = session.store.add(file.filename or "", data, owner=access.user_id)
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
def delete_document(document_id: str, access: Access = Depends(current_access)) -> None:
    session.store.remove(document_id, access.user_id)


@app.get("/api/documents/{document_id}/file")
def document_file(
    document_id: str, access: Access = Depends(current_access)
) -> Response:
    if upload := session.store.get(document_id, access.user_id):
        return Response(
            upload.content,
            media_type=upload.media_type,
            headers={
                "Content-Disposition": f"inline; filename*=UTF-8''{quote(upload.name)}",
                "X-Content-Type-Options": "nosniff",
            },
        )
    # Only IDs of documents the user may read are served, which also rules out path
    # traversal. Foreign documents answer like missing ones.
    group = document_groups().get(document_id)
    if group is None or not access.can_read(group):
        raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
    path = PDF_DIR / group / f"{document_id}.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="PDF nicht vorhanden")
    return FileResponse(path, media_type="application/pdf")


@app.post("/api/chat", response_model_exclude_none=True)
def chat(
    request: ChatRequest, access: Access = Depends(current_access)
) -> ChatResponse:
    answer = answer_with_sources(
        request.question,
        uploaded_document_ids=request.document_ids,
        search_mode=request.search_mode,
        history=request.turns(),
        access=access,
    )
    sources = [_source_out(source) for source in answer.cited_sources()]
    return ChatResponse(answer=answer.text, sources=sources)


@app.post("/api/chat/stream")
def chat_stream(
    request: ChatRequest, access: Access = Depends(current_access)
) -> StreamingResponse:
    """Stream the answer as NDJSON events: sources, delta (repeated), then done or error."""
    prepared = prepare_answer(
        request.question,
        uploaded_document_ids=request.document_ids,
        search_mode=request.search_mode,
        history=request.turns(),
        access=access,
    )

    def events() -> Iterator[str]:
        candidates = [
            _source_out(s).model_dump(exclude_none=True) for s in prepared.sources
        ]
        yield _event("sources", sources=candidates)
        parts: list[str] = []
        try:
            for delta in stream_answer(prepared):
                parts.append(delta)
                yield _event("delta", text=delta)
        except Exception as error:
            logger.exception("Streaming the answer failed")
            yield _event(
                "error", detail=BUSY_MESSAGE if _is_busy(error) else FAILED_MESSAGE
            )
            return
        answer = Answer(text="".join(parts), sources=prepared.sources)
        yield _event("done", cited=[s.number for s in answer.cited_sources()])

    return StreamingResponse(events(), media_type="application/x-ndjson")


def _event(kind: str, **data: object) -> str:
    return json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n"


def _source_out(source: Source) -> SourceOut:
    titles = document_titles()
    return SourceOut(
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
