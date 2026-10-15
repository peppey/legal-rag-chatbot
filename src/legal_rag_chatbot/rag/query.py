import re
import threading
from collections.abc import Collection
from functools import lru_cache
from typing import Any, Literal

import agentplatform
from google import genai
from google.genai import types as genai_types

from legal_rag_chatbot.access import GROUPS, chunk_file_name
from legal_rag_chatbot.evaluation.dataset import load_corpus

DOC_ID_PATTERN = re.compile(r"^document_id (\S+)", re.MULTILINE)

PROJECT_ID = "legal-rag-chatbot"
LOCATION = "europe-west3"

CORPUS_NAME = (
    "projects/187644435346/locations/europe-west3/ragCorpora/7454583283205013504"
)

Ranker = Literal["llm", "service"]

# Quota errors (429) are usually gone after a few seconds, so a chat message retries them
# with backoff (about 1, 2 and 4 seconds) instead of failing at once.
GENAI_RETRY = genai_types.HttpRetryOptions(
    attempts=4, initial_delay=1.0, max_delay=15.0, http_status_codes=[429, 500, 503]
)

# lru_cache alone can build a client twice when threads start together; a discarded
# client closes its connection when it is garbage collected, even mid-request.
_client_lock = threading.Lock()


@lru_cache(maxsize=1)
def _rag_client() -> agentplatform.Client:
    return agentplatform.Client(project=PROJECT_ID, location=LOCATION)


@lru_cache(maxsize=1)
def _genai_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=genai_types.HttpOptions(retry_options=GENAI_RETRY),
    )


def rag_client() -> agentplatform.Client:
    with _client_lock:
        return _rag_client()


def genai_client() -> genai.Client:
    with _client_lock:
        return _genai_client()


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
    *,
    groups: Collection[str],
) -> list[Any]:
    """Return the top_k retrieved contexts from the files of the given groups, best first."""
    client = client or rag_client()
    file_ids = rag_file_ids(groups)
    if not file_ids:
        # Without file ids Vertex would search the whole corpus.
        return []

    response = client.rag.retrieve_contexts(
        vertex_rag_store=genai_types.VertexRagStore(
            rag_resources=[
                genai_types.VertexRagStoreRagResource(
                    rag_corpus=CORPUS_NAME,
                    rag_file_ids=file_ids,
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
    for doc_id, content in (
        corpus if corpus is not None else normalized_corpus()
    ).items():
        if snippet in content:
            return doc_id
    return None


@lru_cache(maxsize=1)
def _rag_file_index() -> dict[str, str]:
    """RagFile display name to file id. Cached, so restart after re-importing the corpus."""
    files = rag_client().rag.list_files(name=CORPUS_NAME).rag_files or []
    return {file.display_name: file.name.rsplit("/", 1)[-1] for file in files}


def rag_file_ids(groups: Collection[str]) -> list[str]:
    """Ids of the RagFiles holding the chunks of the given groups."""
    index = _rag_file_index()
    names = [chunk_file_name(group) for group in GROUPS if group in groups]
    return [index[name] for name in names if name in index]


def retrieve_contexts(question: str) -> None:
    """Retrieve relevant document chunks for a question (all groups, for local checks)."""
    for context in fetch_contexts(question, groups=GROUPS):
        print("=" * 80)
        print(f"Source: {context.source_uri}")
        print(context.text)


if __name__ == "__main__":
    retrieve_contexts("Welche rechtlichen Probleme bestehen in diesem Fall?")
