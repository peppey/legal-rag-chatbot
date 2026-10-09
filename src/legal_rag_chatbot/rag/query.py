from typing import Any

import agentplatform
from google.genai import types as genai_types

PROJECT_ID = "legal-rag-chatbot"
LOCATION = "europe-west3"

CORPUS_NAME = (
    "projects/187644435346/locations/europe-west3/ragCorpora/7991637538768945152"
)


def fetch_contexts(
    question: str,
    top_k: int = 5,
    client: agentplatform.Client | None = None,
) -> list[Any]:
    """Return the top_k retrieved contexts for a question, best first."""
    client = client or agentplatform.Client(
        project=PROJECT_ID,
        location=LOCATION,
    )

    response = client.rag.retrieve_contexts(
        vertex_rag_store=genai_types.VertexRagStore(
            rag_resources=[
                genai_types.VertexRagStoreRagResource(
                    rag_corpus=CORPUS_NAME,
                )
            ],
        ),
        query=agentplatform.types.RagQuery(
            text=question,
            rag_retrieval_config=genai_types.RagRetrievalConfig(
                top_k=top_k,
            ),
        ),
    )

    if response.contexts is None:
        return []
    return list(response.contexts.contexts)


def retrieve_contexts(question: str) -> None:
    """Retrieve relevant document chunks for a question."""
    for context in fetch_contexts(question):
        print("=" * 80)
        print(f"Source: {context.source_uri}")
        print(context.text)


if __name__ == "__main__":
    retrieve_contexts("Welche rechtlichen Probleme bestehen in diesem Fall?")
