import re
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from google.genai import types

from legal_rag_chatbot import prompts
from legal_rag_chatbot.rag import session
from legal_rag_chatbot.rag.query import (
    fetch_contexts,
    genai_client,
    map_chunk_to_document,
)

PROJECT_ID = "legal-rag-chatbot"
LOCATION = "europe-west3"
MODEL = "gemini-2.5-flash"
# Thinking adds about two seconds per answer; the answers are extractive and cited.
THINKING_BUDGET = 0
# Resolves follow-up questions with the chat history before the search (see standalone_question).
REWRITE_FOLLOW_UPS = True

RETRIEVE_K = 20
CONTEXT_K = 5
UPLOAD_CONTEXT_K = 5
# Uploads up to this size are passed in full, which also lets "fasse das Dokument zusammen" work.
UPLOAD_FULL_TEXT_WORDS = 8000
RANKER_MODEL = "semantic-ranker-512@latest"

CORPUS_NAME = (
    "projects/187644435346/locations/europe-west3/ragCorpora/1152921504606846976"
)


# The prompt texts live in prompts/<version>/*.md, the version is set with PROMPT_VERSION.
SYSTEM_PROMPT = f"\n{prompts.get('system')}\n\n{prompts.get('format')}\n"
CITATION_INSTRUCTION = f"\n{prompts.get('citation')}\n"
UPLOAD_INSTRUCTION = f"\n{prompts.get('upload')}\n"
REWRITE_PROMPT = f"\n{prompts.get('rewrite')}\n"

_CORPUS_WORDING = "die aus dem\nRAG-Corpus abgerufen werden"
assert _CORPUS_WORDING in SYSTEM_PROMPT, (
    "system prompt no longer matches the rerank wording"
)
RERANK_SYSTEM_PROMPT = (
    SYSTEM_PROMPT.replace(
        _CORPUS_WORDING, "die als\nDokumentauszüge bereitgestellt werden"
    )
    + CITATION_INSTRUCTION
)
# Only the latest turns matter for resolving references, and answers can be long.
HISTORY_TURNS = 6
HISTORY_TURN_CHARS = 1500
CITATION_MARKER = re.compile(r"\s?\[\d+\]")

# A question with a file number is searchable on its own. Otherwise it is treated as a
# follow-up if it is short, starts like a continuation or contains a reference word.
# Over-matching only costs one extra model call.
FILE_NUMBER = re.compile(r"\b(?:CASE|KANZLEI)-\d+", re.IGNORECASE)
FOLLOW_UP_START = re.compile(
    r"(?:und|aber|oder|auch|noch|dann|also|wie sieht es)\b", re.IGNORECASE
)
REFERENCE_WORD = re.compile(
    r"\b(?:sie|er|es|ihm|ihn|ihnen|ihr\w*|sein\w*|das|dies\w*|jene\w*|dort|dann"
    r"|dafür|dazu|davon|dabei|daraus|darüber|hierzu|hierfür|damit|derselbe|dieselbe)\b",
    re.IGNORECASE,
)
MIN_STANDALONE_WORDS = 7


def needs_rewrite(question: str) -> bool:
    """Heuristic: does the question probably depend on the previous chat messages?"""
    if FILE_NUMBER.search(question):
        return False
    return (
        len(question.split()) < MIN_STANDALONE_WORDS
        or bool(FOLLOW_UP_START.match(question.strip()))
        or bool(REFERENCE_WORD.search(question))
    )


@dataclass
class Turn:
    """One earlier chat message; role is "user" or "assistant"."""

    role: str
    content: str


@dataclass
class Source:
    """A retrieved excerpt; number matches the [n] markers in the answer."""

    number: int
    document_id: str | None
    source_uri: str | None
    text: str
    # Set only for excerpts of uploaded session documents.
    document_name: str | None = None
    page: int | None = None


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)

    def cited_sources(self) -> list[Source]:
        """Sources actually referenced as [n] in the answer text."""
        return [s for s in self.sources if f"[{s.number}]" in self.text]


@dataclass
class PreparedAnswer:
    """Retrieved excerpts and the Gemini request built from them."""

    sources: list[Source]
    system_prompt: str
    contents: str

    @property
    def config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        )


def standalone_question(question: str, history: Sequence[Turn]) -> str:
    """Rewrite a follow-up question so that it is understandable without the chat history."""
    turns = history[-HISTORY_TURNS:]
    if not REWRITE_FOLLOW_UPS or not turns or not needs_rewrite(question):
        return question
    transcript = "\n".join(
        f"{'Nutzer' if turn.role == 'user' else 'Assistent'}: "
        f"{CITATION_MARKER.sub('', turn.content)[:HISTORY_TURN_CHARS]}"
        for turn in turns
    )
    response = genai_client().models.generate_content(
        model=MODEL,
        contents=f"Gesprächsverlauf:\n{transcript}\n\nLetzte Frage: {question}",
        config=types.GenerateContentConfig(
            system_instruction=REWRITE_PROMPT,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip() or question


def prepare_answer(
    question: str,
    retrieve_k: int = RETRIEVE_K,
    context_k: int = CONTEXT_K,
    uploaded_document_ids: Sequence[str] = (),
    history: Sequence[Turn] = (),
) -> PreparedAnswer:
    """Retrieve and rerank retrieve_k chunks and collect the best context_k as excerpts.

    Excerpts of the given uploaded session documents are added in front of the corpus
    excerpts, they are searched separately and never part of the RAG corpus. Follow-up
    questions are rewritten with the chat history first, so that the search finds them.
    """
    question = standalone_question(question, history)
    with ThreadPoolExecutor(max_workers=1) as pool:
        corpus_search = pool.submit(
            fetch_contexts,
            question,
            top_k=retrieve_k,
            ranker="service",
            ranker_model=RANKER_MODEL,
        )
        hits = session.store.full_text(
            uploaded_document_ids, UPLOAD_FULL_TEXT_WORDS
        ) or session.store.search(question, uploaded_document_ids, UPLOAD_CONTEXT_K)
        contexts = corpus_search.result()[:context_k]

    sources = [
        Source(
            number=number,
            document_id=hit.document_id,
            source_uri=None,
            text=hit.text,
            document_name=hit.document_name,
            page=hit.page,
        )
        for number, hit in enumerate(hits, start=1)
    ]
    sources += [
        Source(
            number=number,
            document_id=map_chunk_to_document(c.text),
            source_uri=c.source_uri,
            text=c.text,
        )
        for number, c in enumerate(contexts, start=len(hits) + 1)
    ]
    excerpts = "\n\n".join(f"[{s.number}] {_excerpt_label(s)}{s.text}" for s in sources)
    return PreparedAnswer(
        sources=sources,
        system_prompt=RERANK_SYSTEM_PROMPT + (UPLOAD_INSTRUCTION if hits else ""),
        contents=f"Dokumentauszüge:\n\n{excerpts}\n\nFrage: {question}",
    )


def answer_with_sources(
    question: str,
    retrieve_k: int = RETRIEVE_K,
    context_k: int = CONTEXT_K,
    uploaded_document_ids: Sequence[str] = (),
    history: Sequence[Turn] = (),
) -> Answer:
    """Retrieve and rerank retrieve_k chunks, then answer from the best context_k."""
    prepared = prepare_answer(
        question, retrieve_k, context_k, uploaded_document_ids, history
    )
    response = genai_client().models.generate_content(
        model=MODEL, contents=prepared.contents, config=prepared.config
    )
    return Answer(text=response.text or "", sources=prepared.sources)


def stream_answer(prepared: PreparedAnswer) -> Iterator[str]:
    """Yield the answer text in pieces as Gemini generates it."""
    for chunk in genai_client().models.generate_content_stream(
        model=MODEL, contents=prepared.contents, config=prepared.config
    ):
        if chunk.text:
            yield chunk.text


def _excerpt_label(source: Source) -> str:
    if source.document_name is None:
        return ""
    page = f", Seite {source.page}" if source.page else ""
    return f"Hochgeladenes Dokument „{source.document_name}“{page}: "


def answer_question(
    question: str,
    retrieve_k: int = RETRIEVE_K,
    context_k: int = CONTEXT_K,
    history: Sequence[Turn] = (),
) -> str:
    """Like answer_with_sources, but returns only the answer text with [n] markers."""
    return answer_with_sources(question, retrieve_k, context_k, history=history).text


def answer_question_grounded(question: str) -> str:
    """Answer a question with Gemini's built-in RAG grounding tool (top_k=5)."""

    client = genai_client()

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
