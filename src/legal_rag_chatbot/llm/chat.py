from google import genai
from google.genai import types

PROJECT_ID = "legal-rag-chatbot"
LOCATION = "europe-west3"

CORPUS_NAME = (
    "projects/legal-rag-chatbot/locations/europe-west3/ragCorpora/7991637538768945152"
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


def answer_question(question: str) -> str:
    """Answer a question with Gemini, grounded in the RAG corpus."""

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
        model="gemini-2.5-flash",
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
