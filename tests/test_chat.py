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
