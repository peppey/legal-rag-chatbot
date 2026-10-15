from types import SimpleNamespace

import pytest

from legal_rag_chatbot.access import SYSTEM_ACCESS
from legal_rag_chatbot.llm import chat
from legal_rag_chatbot.llm.chat import Turn


class FakeClient:
    def __init__(self, reply: str) -> None:
        self.calls: list[str] = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self._reply = reply

    def _generate(self, *, model, contents, config):
        self.calls.append(contents)
        return SimpleNamespace(text=self._reply)


def test_standalone_question_skips_the_model_without_history() -> None:
    assert chat.standalone_question("Wie hoch ist die Miete?", []) == (
        "Wie hoch ist die Miete?"
    )


def test_standalone_question_rewrites_with_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient("Bis wann muss Frau Keller dem Vermieter widersprechen?  ")
    monkeypatch.setattr(chat, "genai_client", lambda: client)
    history = [
        Turn("user", "Wer ist Frau Keller?"),
        Turn("assistant", "Frau Keller ist die Mandantin in CASE-001 [1]."),
    ]

    result = chat.standalone_question("Und bis wann muss sie widersprechen?", history)

    assert result == "Bis wann muss Frau Keller dem Vermieter widersprechen?"
    assert "Nutzer: Wer ist Frau Keller?" in client.calls[0]
    assert "Assistent: Frau Keller ist die Mandantin in CASE-001." in client.calls[0]
    assert "Letzte Frage: Und bis wann muss sie widersprechen?" in client.calls[0]


def test_standalone_question_keeps_the_question_if_the_model_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat, "genai_client", lambda: FakeClient(""))

    assert chat.standalone_question("Und dann?", [Turn("user", "Hallo")]) == "Und dann?"


@pytest.mark.parametrize(
    "question",
    [
        "Und hat sie das schon getan?",
        "Wann hat er sie dann eingereicht?",
        "Wie hoch ist die Miete?",
        "Welche Frist gilt dafür, wenn der Vermieter die Kündigung ausgesprochen hat?",
    ],
)
def test_needs_rewrite_for_follow_ups(question: str) -> None:
    assert chat.needs_rewrite(question)


@pytest.mark.parametrize(
    "question",
    [
        "Bis wann muss Frau Keller der Eigenbedarfskündigung des Vermieters Wendt widersprechen?",
        "Was ist im Fall CASE-002 die empfohlene Abfindung?",
        "Welche Gebühren sieht die Gebührenvereinbarung der Kanzlei für Beratungen vor?",
    ],
)
def test_standalone_questions_are_not_rewritten(
    question: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not chat.needs_rewrite(question)

    def fail():
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(chat, "genai_client", fail)
    assert chat.standalone_question(question, [Turn("user", "Hallo")]) == question


def test_genai_client_is_created_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    from legal_rag_chatbot.rag import query

    created: list[object] = []

    def slow_client(**kwargs):
        time.sleep(0.05)
        created.append(object())
        return created[-1]

    query._genai_client.cache_clear()
    monkeypatch.setattr(query.genai, "Client", slow_client)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            clients = list(pool.map(lambda _: query.genai_client(), range(8)))
    finally:
        query._genai_client.cache_clear()

    assert len(created) == 1
    assert all(client is created[0] for client in clients)


def test_standalone_question_is_skipped_when_rewriting_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail():
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(chat, "REWRITE_FOLLOW_UPS", False)
    monkeypatch.setattr(chat, "genai_client", fail)

    assert chat.standalone_question("Und dann?", [Turn("user", "Hallo")]) == "Und dann?"


@pytest.mark.parametrize(
    "question", ["Hallo", "Danke!", "Vielen Dank, tschüss", "Wer bist du?"]
)
def test_smalltalk_is_detected(question: str) -> None:
    assert chat.is_smalltalk(question)


@pytest.mark.parametrize(
    "question",
    [
        "Hallo, wie hoch ist die Miete in CASE-005?",
        "Und hat sie das schon getan?",
        "Danke, und wie hoch ist die Abfindung?",
        "???",
    ],
)
def test_real_questions_are_not_smalltalk(question: str) -> None:
    assert not chat.is_smalltalk(question)


def test_heuristic_route() -> None:
    assert chat.heuristic_route("Hallo", has_uploads=True) is chat.Route.NONE
    assert chat.heuristic_route("Wie hoch ist die Miete?", False) is chat.Route.CORPUS
    assert chat.heuristic_route("Was steht in CASE-002?", True) is chat.Route.BOTH
    assert chat.heuristic_route("Fasse das zusammen", True) is None


def _fail(*args, **kwargs):
    raise AssertionError("must not be called")


def test_smalltalk_needs_no_search_and_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat, "fetch_contexts", _fail)
    monkeypatch.setattr(chat, "genai_client", _fail)

    prepared = chat.prepare_answer("Hallo", search_mode="auto", access=SYSTEM_ACCESS)

    assert prepared.route is chat.Route.NONE
    assert prepared.sources == []
    assert prepared.system_prompt == chat.SMALLTALK_SYSTEM_PROMPT
    assert prepared.contents == "Hallo"


def _hit(text: str = "Vertragstext"):
    return SimpleNamespace(
        document_id="upload-1", document_name="vertrag.txt", page=1, text=text
    )


def test_upload_route_skips_the_corpus_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "fetch_contexts", _fail)
    monkeypatch.setattr(
        chat.session.store, "full_text", lambda ids, words, **kwargs: [_hit()]
    )

    prepared = chat.prepare_answer(
        "Fasse das Dokument zusammen",
        uploaded_document_ids=["upload-1"],
        search_mode="uploads",
        access=SYSTEM_ACCESS,
    )

    assert prepared.route is chat.Route.UPLOADS
    assert [s.document_name for s in prepared.sources] == ["vertrag.txt"]
    assert chat.UPLOAD_INSTRUCTION in prepared.system_prompt


def test_upload_route_falls_back_to_the_corpus_without_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searched: list[str] = []
    monkeypatch.setattr(chat.session.store, "full_text", lambda ids, words, **kw: [])
    monkeypatch.setattr(chat.session.store, "search", lambda q, ids, k, **kw: [])
    monkeypatch.setattr(
        chat, "fetch_contexts", lambda q, **kwargs: searched.append(q) or []
    )

    chat.prepare_answer(
        "Wie hoch ist die Miete?", search_mode="uploads", access=SYSTEM_ACCESS
    )

    assert searched == ["Wie hoch ist die Miete?"]


def test_corpus_route_ignores_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "fetch_contexts", lambda q, **kwargs: [])
    monkeypatch.setattr(chat.session.store, "full_text", _fail)
    monkeypatch.setattr(chat.session.store, "search", _fail)

    prepared = chat.prepare_answer(
        "Wie hoch ist die Miete?",
        uploaded_document_ids=["upload-1"],
        search_mode="corpus",
        access=SYSTEM_ACCESS,
    )

    assert prepared.route is chat.Route.CORPUS
    assert chat.UPLOAD_INSTRUCTION not in prepared.system_prompt


def test_decide_route_uses_the_model_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = chat.RouteDecision(
        question="Fasse vertrag.txt zusammen", route="uploads"
    )
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(parsed=decision)
        )
    )
    monkeypatch.setattr(chat, "genai_client", lambda: client)

    question, route = chat.decide_route("Fasse das zusammen", [], ["vertrag.txt"])

    assert route is chat.Route.UPLOADS
    # Without history the question is kept, the model only picks the route.
    assert question == "Fasse das zusammen"


def test_decide_route_searches_everything_if_the_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(chat, "genai_client", broken)

    assert chat.decide_route("Fasse das zusammen", [], ["a.txt"]) == (
        "Fasse das zusammen",
        chat.Route.BOTH,
    )


def test_genai_client_retries_quota_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from legal_rag_chatbot.rag import query

    captured: dict = {}
    monkeypatch.setattr(query.genai, "Client", lambda **kwargs: captured.update(kwargs))
    query._genai_client.cache_clear()
    try:
        query.genai_client()
    finally:
        query._genai_client.cache_clear()

    retry = captured["http_options"].retry_options
    assert 429 in retry.http_status_codes
    assert retry.attempts > 1
