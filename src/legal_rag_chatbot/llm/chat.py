import logging
import re
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from google.genai import types
from pydantic import BaseModel

from legal_rag_chatbot import prompts
from legal_rag_chatbot.access import SYSTEM_ACCESS, Access
from legal_rag_chatbot.rag import session
from legal_rag_chatbot.rag.query import (
    fetch_contexts,
    genai_client,
    map_chunk_to_document,
    rag_file_ids,
)

logger = logging.getLogger(__name__)

PROJECT_ID = "legal-rag-chatbot"
LOCATION = "europe-west3"
MODEL = "gemini-2.5-flash"
# Thinking adds about two seconds per answer; the answers are extractive and cited.
THINKING_BUDGET = 0
# With the thinking mode on the model decides how much to think (-1 = dynamic).
THINKING_BUDGET_ON = -1
# Resolves follow-up questions with the chat history before the search (see standalone_question).
REWRITE_FOLLOW_UPS = True

RETRIEVE_K = 20
CONTEXT_K = 5
UPLOAD_CONTEXT_K = 5
# Uploads up to this size are passed in full, which also lets "fasse das Dokument zusammen" work.
UPLOAD_FULL_TEXT_WORDS = 8000
RANKER_MODEL = "semantic-ranker-512@latest"

CORPUS_NAME = (
    "projects/187644435346/locations/europe-west3/ragCorpora/7454583283205013504"
)


# The prompt texts live in prompts/<version>/*.md, the version is set with PROMPT_VERSION.
SYSTEM_PROMPT = f"\n{prompts.get('system')}\n\n{prompts.get('format')}\n"
CITATION_INSTRUCTION = f"\n{prompts.get('citation')}\n"
UPLOAD_INSTRUCTION = f"\n{prompts.get('upload')}\n"
REWRITE_PROMPT = f"\n{prompts.get('rewrite')}\n"
ROUTE_PROMPT = f"\n{prompts.get('route')}\n"
SMALLTALK_SYSTEM_PROMPT = f"\n{prompts.get('smalltalk')}\n"

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


class Route(str, Enum):
    """Which sources a question is answered from."""

    NONE = "none"
    UPLOADS = "uploads"
    CORPUS = "corpus"
    BOTH = "both"


# "auto" decides per question, the others force a route. Evaluations use "corpus".
SearchMode = Literal["auto", "none", "uploads", "corpus"]

SMALLTALK_WORDS = frozenset(
    "hallo hi hey moin servus guten tag morgen abend danke dankeschön schön sehr "
    "vielen herzlichen dank tschüss tschüs wiedersehen ciao ok okay alles klar "
    "super prima gut verstanden perfekt gern gerne".split()
)
ASSISTANT_QUESTION = re.compile(
    r"\b(?:wer bist du|was kannst du|was machst du|wie funktionierst du)\b",
    re.IGNORECASE,
)


def is_smalltalk(question: str) -> bool:
    """Greetings, thanks and questions about the assistant itself need no documents."""
    if FILE_NUMBER.search(question):
        return False
    words = re.findall(r"[a-zäöüß]+", question.lower())
    if not words:
        return False
    return all(w in SMALLTALK_WORDS for w in words) or (
        len(words) <= 8 and bool(ASSISTANT_QUESTION.search(question))
    )


def heuristic_route(question: str, has_uploads: bool) -> Route | None:
    """Route without a model call, or None if only the model can tell (uploads present)."""
    if is_smalltalk(question):
        return Route.NONE
    if not has_uploads:
        return Route.CORPUS
    if FILE_NUMBER.search(question):
        return Route.BOTH
    return None


class RouteDecision(BaseModel):
    question: str
    route: Literal["none", "uploads", "corpus", "both"]


def decide_route(
    question: str, history: "Sequence[Turn]", upload_names: Sequence[str]
) -> tuple[str, Route]:
    """One model call that picks the route and rewrites follow-up questions.

    On any failure the question is answered from everything, which is the safe default.
    """
    turns = history[-HISTORY_TURNS:]
    parts = [f"Hochgeladene Dokumente: {', '.join(upload_names) or '(keine)'}"]
    if turns:
        parts.append(f"Gesprächsverlauf:\n{_transcript(turns)}")
    parts.append(f"Letzte Frage: {question}")
    try:
        response = genai_client().models.generate_content(
            model=MODEL,
            contents="\n\n".join(parts),
            config=types.GenerateContentConfig(
                system_instruction=ROUTE_PROMPT,
                temperature=0,
                response_mime_type="application/json",
                response_schema=RouteDecision,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        decision = response.parsed
        route = Route(decision.route)
    except Exception:
        logger.exception("Routing failed, searching all sources")
        return question, Route.BOTH
    rewrite = bool(turns) and REWRITE_FOLLOW_UPS and needs_rewrite(question)
    return (decision.question.strip() or question if rewrite else question), route


def resolve_route(
    question: str,
    history: "Sequence[Turn]",
    uploaded_document_ids: Sequence[str],
    search_mode: SearchMode,
    access: Access,
) -> tuple[str, Route]:
    """Standalone question and route for a chat message."""
    if search_mode != "auto":
        return standalone_question(question, history), Route(search_mode)
    route = heuristic_route(question, bool(uploaded_document_ids))
    if route is None:
        names = [
            d.name
            for i in uploaded_document_ids
            if (d := session.store.get(i, access.user_id))
        ]
        return decide_route(question, history, names)
    if route is Route.NONE:
        return question, route
    return standalone_question(question, history), route


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
    route: Route = Route.CORPUS
    thinking: bool = False

    @property
    def config(self) -> types.GenerateContentConfig:
        budget = THINKING_BUDGET_ON if self.thinking else THINKING_BUDGET
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=budget),
        )


def _transcript(turns: Sequence[Turn]) -> str:
    return "\n".join(
        f"{'Nutzer' if turn.role == 'user' else 'Assistent'}: "
        f"{CITATION_MARKER.sub('', turn.content)[:HISTORY_TURN_CHARS]}"
        for turn in turns
    )


def standalone_question(question: str, history: Sequence[Turn]) -> str:
    """Rewrite a follow-up question so that it is understandable without the chat history."""
    turns = history[-HISTORY_TURNS:]
    if not REWRITE_FOLLOW_UPS or not turns or not needs_rewrite(question):
        return question
    response = genai_client().models.generate_content(
        model=MODEL,
        contents=f"Gesprächsverlauf:\n{_transcript(turns)}\n\nLetzte Frage: {question}",
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
    search_mode: SearchMode = "corpus",
    thinking: bool = False,
    *,
    access: Access,
) -> PreparedAnswer:
    """Retrieve and rerank retrieve_k chunks and collect the best context_k as excerpts.

    Only documents of the groups in access are searched. Excerpts of the given uploaded
    session documents (of the same user) are added in front of the corpus
    excerpts, they are searched separately and never part of the RAG corpus. Follow-up
    questions are rewritten with the chat history first, so that the search finds them.
    The route (see resolve_route) decides which of the two searches run at all.
    """
    question, route = resolve_route(
        question, history, uploaded_document_ids, search_mode, access
    )
    logger.info("Answering with route %s", route.value)
    if route is Route.NONE:
        return PreparedAnswer(
            sources=[],
            system_prompt=SMALLTALK_SYSTEM_PROMPT,
            contents=question,
            route=route,
        )

    search_corpus = route in (Route.CORPUS, Route.BOTH)
    search_uploads = route in (Route.UPLOADS, Route.BOTH)
    with ThreadPoolExecutor(max_workers=1) as pool:

        def start_corpus_search():
            return pool.submit(
                fetch_contexts,
                question,
                top_k=retrieve_k,
                ranker="service",
                ranker_model=RANKER_MODEL,
                groups=access.groups,
            )

        corpus_search = start_corpus_search() if search_corpus else None
        hits = (
            session.store.full_text(
                uploaded_document_ids, UPLOAD_FULL_TEXT_WORDS, owner=access.user_id
            )
            or session.store.search(
                question,
                uploaded_document_ids,
                UPLOAD_CONTEXT_K,
                owner=access.user_id,
            )
            if search_uploads
            else []
        )
        if corpus_search is None and not hits:
            # Nothing found in the uploads, so fall back to the corpus.
            corpus_search = start_corpus_search()
        contexts = corpus_search.result()[:context_k] if corpus_search else []

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
        route=route,
        thinking=thinking,
    )


def answer_with_sources(
    question: str,
    retrieve_k: int = RETRIEVE_K,
    context_k: int = CONTEXT_K,
    uploaded_document_ids: Sequence[str] = (),
    history: Sequence[Turn] = (),
    search_mode: SearchMode = "corpus",
    thinking: bool = False,
    *,
    access: Access,
) -> Answer:
    """Retrieve and rerank retrieve_k chunks, then answer from the best context_k."""
    prepared = prepare_answer(
        question,
        retrieve_k,
        context_k,
        uploaded_document_ids,
        history,
        search_mode,
        thinking,
        access=access,
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
    *,
    access: Access,
) -> str:
    """Like answer_with_sources, but returns only the answer text with [n] markers."""
    return answer_with_sources(
        question, retrieve_k, context_k, history=history, access=access
    ).text


def answer_question_grounded(question: str, *, access: Access) -> str:
    """Answer a question with Gemini's built-in RAG grounding tool (top_k=5)."""

    file_ids = rag_file_ids(access.groups)
    if not file_ids:
        # Without file ids the tool would search the whole corpus.
        raise PermissionError("No readable documents for this user")
    client = genai_client()

    rag_tool = types.Tool(
        retrieval=types.Retrieval(
            vertex_rag_store=types.VertexRagStore(
                rag_resources=[
                    types.VertexRagStoreRagResource(
                        rag_corpus=CORPUS_NAME,
                        rag_file_ids=file_ids,
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
        "Welche rechtlichen Probleme bestehen im Fall CASE-024?",
        access=SYSTEM_ACCESS,
    )

    print(answer.text)
    print("\nQuellen:")
    for source in answer.cited_sources():
        print(f"[{source.number}] {source.document_id or '?'} ({source.source_uri})")
