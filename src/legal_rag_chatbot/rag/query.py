import re
from functools import lru_cache
from typing import Any, Literal

import agentplatform
from google.genai import types as genai_types

from legal_rag_chatbot.evaluation.dataset import load_corpus

DOC_ID_PATTERN = re.compile(r"^document_id (\S+)", re.MULTILINE)

PROJECT_ID = "legal-rag-chatbot"
LOCATION = "europe-west3"

CORPUS_NAME = (
    "projects/legal-rag-chatbot/locations/europe-west3/ragCorpora/4532873024948404224"
)

Ranker = Literal["llm", "service"]


def build_retrieval_config(
    top_k: int,
    ranker: Ranker | None = None,
    ranker_model: str | None = None,
) -> genai_types.RagRetrievalConfig:
    """Build the retrieval config, optionally with a reranker (llm or rank service)."""
    ranking = None
    if ranker == "llm":
        ranking = genai_types.RagRetrievalConfigRanking(
            llm_ranker=genai_types.RagRetrievalConfigRankingLlmRanker(
                model_name=ranker_model
            )
        )
    elif ranker == "service":
        ranking = genai_types.RagRetrievalConfigRanking(
            rank_service=genai_types.RagRetrievalConfigRankingRankService(
                model_name=ranker_model
            )
        )
    return genai_types.RagRetrievalConfig(top_k=top_k, ranking=ranking)


def fetch_contexts(
    question: str,
    top_k: int = 5,
    client: agentplatform.Client | None = None,
    ranker: Ranker | None = None,
    ranker_model: str | None = None,
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
            rag_retrieval_config=build_retrieval_config(top_k, ranker, ranker_model),
        ),
    )

    if response.contexts is None:
        return []
    return list(response.contexts.contexts)


def normalize(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return " ".join(text.split())


@lru_cache(maxsize=1)
def normalized_corpus() -> dict[str, str]:
    """Local document contents keyed by document_id, whitespace-normalized."""
    return {doc_id: normalize(doc["content"]) for doc_id, doc in load_corpus().items()}


def map_chunk_to_document(
    chunk_text: str, corpus: dict[str, str] | None = None
) -> str | None:
    """Resolve a retrieved chunk to a document_id (defaults to the local corpus)."""
    match = DOC_ID_PATTERN.search(chunk_text)
    if match:
        return match.group(1)
    # Follow-up chunks of long documents carry no header; match by content.
    snippet = normalize(chunk_text)[:80]
    if not snippet:
        return None
    for doc_id, content in (corpus if corpus is not None else normalized_corpus()).items():
        if snippet in content:
            return doc_id
    return None


def retrieve_contexts(question: str) -> None:
    """Retrieve relevant document chunks for a question."""
    for context in fetch_contexts(question):
        print("=" * 80)
        print(f"Source: {context.source_uri}")
        print(context.text)


if __name__ == "__main__":
    retrieve_contexts("Welche rechtlichen Probleme bestehen in diesem Fall?")
