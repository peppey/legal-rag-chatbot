from google import genai
from google.genai import types

from legal_rag_chatbot.rag.query import fetch_contexts

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

RERANK_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "die aus dem\nRAG-Corpus abgerufen werden",
    "die als\nDokumentauszüge bereitgestellt werden",
)


def answer_question(
    question: str, retrieve_k: int = RETRIEVE_K, context_k: int = CONTEXT_K
) -> str:
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
    return response.text


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
    answer = answer_question("Welche rechtlichen Probleme bestehen im Fall CASE-024?")

    print(answer)
