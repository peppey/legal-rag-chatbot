from dataclasses import dataclass, field

from google import genai
from google.genai import types

from legal_rag_chatbot.rag.query import fetch_contexts, map_chunk_to_document

PROJECT_ID = "legal-rag-chatbot"
LOCATION = "europe-west3"
MODEL = "gemini-2.5-flash"

RETRIEVE_K = 20
CONTEXT_K = 5
RANKER_MODEL = "semantic-ranker-512@latest"

CORPUS_NAME = (
    "projects/legal-rag-chatbot/locations/europe-west3/ragCorpora/4532873024948404224"
)


SYSTEM_PROMPT = """
Du bist ein juristischer Rechercheassistent für eine deutsche Kanzlei.

Beantworte Fragen auf Grundlage der Informationen, die aus dem
RAG-Corpus abgerufen werden.

Erfinde keine Tatsachen. Wenn die vorhandenen Dokumente keine
ausreichende Grundlage für eine Antwort liefern, sage das ausdrücklich.

Unterscheide zwischen:
- Tatsachen aus den Akten
- rechtlichen Fragestellungen
- rechtlicher Bewertung

Gib keine abschließende Rechtsberatung.
"""

CITATION_INSTRUCTION = """
Belege jede Tatsachenaussage mit der Nummer des Dokumentauszugs in eckigen
Klammern, z. B. [1] oder [2][3]. Verwende nur Nummern der bereitgestellten
Auszüge und zitiere nichts, was dort nicht steht.
"""

RERANK_SYSTEM_PROMPT = (
    SYSTEM_PROMPT.replace(
        "die aus dem\nRAG-Corpus abgerufen werden",
        "die als\nDokumentauszüge bereitgestellt werden",
    )
    + CITATION_INSTRUCTION
)


@dataclass
class Source:
    """A retrieved excerpt; number matches the [n] markers in the answer."""

    number: int
    document_id: str | None
    source_uri: str | None
    text: str


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)

    def cited_sources(self) -> list[Source]:
        """Sources actually referenced as [n] in the answer text."""
        return [s for s in self.sources if f"[{s.number}]" in self.text]


def answer_with_sources(
    question: str, retrieve_k: int = RETRIEVE_K, context_k: int = CONTEXT_K
) -> Answer:
    """Retrieve and rerank retrieve_k chunks, then answer from the best context_k."""
    contexts = fetch_contexts(
        question,
        top_k=retrieve_k,
        ranker="service",
        ranker_model=RANKER_MODEL,
    )[:context_k]
    excerpts = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(contexts, start=1))

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Dokumentauszüge:\n\n{excerpts}\n\nFrage: {question}",
        config=types.GenerateContentConfig(system_instruction=RERANK_SYSTEM_PROMPT),
    )
    sources = [
        Source(
            number=number,
            document_id=map_chunk_to_document(c.text),
            source_uri=c.source_uri,
            text=c.text,
        )
        for number, c in enumerate(contexts, start=1)
    ]
    return Answer(text=response.text or "", sources=sources)


def answer_question(
    question: str, retrieve_k: int = RETRIEVE_K, context_k: int = CONTEXT_K
) -> str:
    """Like answer_with_sources, but returns only the answer text with [n] markers."""
    return answer_with_sources(question, retrieve_k, context_k).text


def answer_question_grounded(question: str) -> str:
    """Answer a question with Gemini's built-in RAG grounding tool (top_k=5)."""

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

    rag_tool = types.Tool(
        retrieval=types.Retrieval(
            vertex_rag_store=types.VertexRagStore(
                rag_resources=[
                    types.VertexRagStoreRagResource(
                        rag_corpus=CORPUS_NAME,
                    )
                ],
                rag_retrieval_config=types.RagRetrievalConfig(
                    top_k=5,
                ),
            ),
        )
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[rag_tool],
        ),
    )

    return response.text


if __name__ == "__main__":
    answer = answer_with_sources(
        "Welche rechtlichen Probleme bestehen im Fall CASE-024?"
    )

    print(answer.text)
    print("\nQuellen:")
    for source in answer.cited_sources():
        print(f"[{source.number}] {source.document_id or '?'} ({source.source_uri})")
