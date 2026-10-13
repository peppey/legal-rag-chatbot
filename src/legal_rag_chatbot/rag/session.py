"""In-memory vector store for documents uploaded during a session.

Uploads are chunked and embedded here only. They are never written to Cloud Storage
or imported into the Vertex RAG corpus, and they are gone when the server stops.
"""

import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from legal_rag_chatbot.ingestion.chunking import chunk_paragraphs
from legal_rag_chatbot.ingestion.extract import ExtractionError, extract_text_units
from legal_rag_chatbot.rag.query import LOCATION, PROJECT_ID

EMBEDDING_MODEL = "text-embedding-005"
EMBED_BATCH_SIZE = 16
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain; charset=utf-8",
}

# Takes texts and a Vertex task type, returns one embedding row per text.
Embedder = Callable[[list[str], str], np.ndarray]


def embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    """Embed texts with Vertex AI and return L2-normalized row vectors."""
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    config = types.EmbedContentConfig(task_type=task_type)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts[start : start + EMBED_BATCH_SIZE],
            config=config,
        )
        vectors.extend(e.values for e in response.embeddings or [])
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@dataclass
class SessionChunk:
    text: str
    page: int | None


@dataclass
class SessionDocument:
    id: str
    name: str
    media_type: str
    content: bytes
    uploaded_at: str
    chunks: list[SessionChunk]
    embeddings: np.ndarray


@dataclass
class SessionHit:
    document_id: str
    document_name: str
    page: int | None
    text: str
    score: float


class SessionStore:
    def __init__(self, embedder: Embedder = embed_texts) -> None:
        self._embedder = embedder
        self._documents: dict[str, SessionDocument] = {}
        self._lock = threading.Lock()

    def add(self, name: str, data: bytes) -> SessionDocument:
        """Extract, chunk and embed an upload. Raises ExtractionError for unreadable files."""
        name = Path(name).name or "Dokument"
        chunks = [
            SessionChunk(text=chunk["content"], page=page)
            for page, paragraphs in extract_text_units(name, data)
            for chunk in chunk_paragraphs(paragraphs)
        ]
        if not chunks:
            raise ExtractionError("Im Dokument wurde kein Text gefunden.")
        document = SessionDocument(
            id=uuid.uuid4().hex,
            name=name,
            media_type=MEDIA_TYPES[Path(name).suffix.lower()],
            content=data,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            chunks=chunks,
            embeddings=self._embedder([c.text for c in chunks], "RETRIEVAL_DOCUMENT"),
        )
        with self._lock:
            self._documents[document.id] = document
        return document

    def get(self, document_id: str) -> SessionDocument | None:
        return self._documents.get(document_id)

    def remove(self, document_id: str) -> bool:
        with self._lock:
            return self._documents.pop(document_id, None) is not None

    def full_text(
        self, document_ids: Iterable[str], max_words: int
    ) -> list[SessionHit]:
        """Return all chunks of the given uploads, or [] if they exceed max_words together."""
        documents = [d for i in dict.fromkeys(document_ids) if (d := self.get(i))]
        words = sum(len(c.text.split()) for d in documents for c in d.chunks)
        if words > max_words:
            return []
        return [
            SessionHit(d.id, d.name, c.page, c.text, score=1.0)
            for d in documents
            for c in d.chunks
        ]

    def search(
        self, question: str, document_ids: Iterable[str], top_k: int
    ) -> list[SessionHit]:
        """Return the chunks of the given uploads that are most similar to the question."""
        documents = [d for i in dict.fromkeys(document_ids) if (d := self.get(i))]
        if not documents:
            return []
        query = self._embedder([question], "RETRIEVAL_QUERY")[0]
        embeddings = np.vstack([d.embeddings for d in documents])
        scores = embeddings @ query
        owners = [(d, c) for d in documents for c in d.chunks]
        best = np.argsort(scores)[::-1][:top_k]
        return [
            SessionHit(
                document_id=owners[i][0].id,
                document_name=owners[i][0].name,
                page=owners[i][1].page,
                text=owners[i][1].text,
                score=float(scores[i]),
            )
            for i in best
        ]


store = SessionStore()
