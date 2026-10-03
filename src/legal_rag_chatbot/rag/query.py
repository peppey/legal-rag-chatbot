import agentplatform
from google.genai import types as genai_types


PROJECT_ID = "legal-rag-chatbot"
LOCATION = "europe-west3"

CORPUS_NAME = (
    "projects/legal-rag-chatbot/locations/europe-west3/"
    "ragCorpora/6917529027641081856"
)


def retrieve_contexts(question: str) -> None:
    """Retrieve relevant document chunks for a question."""
    client = agentplatform.Client(
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
                top_k=5,
            ),
        ),
    )

    for context in response.contexts.contexts:
        print("=" * 80)
        print(f"Source: {context.source_uri}")
        print(context.text)


if __name__ == "__main__":
    retrieve_contexts(
        "Welche rechtlichen Probleme bestehen in diesem Fall?"
    )