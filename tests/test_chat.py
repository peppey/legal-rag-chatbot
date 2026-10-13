from types import SimpleNamespace

import pytest

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
