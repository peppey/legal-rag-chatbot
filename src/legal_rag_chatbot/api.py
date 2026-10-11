from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from legal_rag_chatbot.evaluation.dataset import load_corpus
from legal_rag_chatbot.llm.chat import answer_with_sources

EXCERPT_CHARS = 300

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
