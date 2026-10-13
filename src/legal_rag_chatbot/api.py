from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from legal_rag_chatbot.evaluation.dataset import load_corpus
from legal_rag_chatbot.llm.chat import answer_with_sources

EXCERPT_CHARS = 300
PDF_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "pdf"

app = FastAPI(title="legal-rag-chatbot")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class SourceOut(BaseModel):
    documentId: str
    documentName: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@lru_cache(maxsize=1)
def document_titles() -> dict[str, str]:
    return {doc_id: doc["document_title"] for doc_id, doc in load_corpus().items()}


@app.get("/api/documents/{document_id}/file")
def document_file(document_id: str) -> FileResponse:
    # Only IDs from the corpus are served, which rules out path traversal.
    if document_id not in document_titles():
        raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
    path = PDF_DIR / f"{document_id}.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="PDF nicht vorhanden")
    return FileResponse(path, media_type="application/pdf")


@app.post("/api/chat")
def chat(request: ChatRequest) -> ChatResponse:
    answer = answer_with_sources(request.question)
    titles = document_titles()
    sources = [
        SourceOut(
            documentId=source.document_id or "",
            documentName=(
                f"{source.document_id} – {titles[source.document_id]}"
                if source.document_id in titles
                else "Unbekanntes Dokument"
            ),
            excerpt=f"[{source.number}] {source.text[:EXCERPT_CHARS]}",
        )
        for source in answer.cited_sources()
    ]
    return ChatResponse(answer=answer.text, sources=sources)
