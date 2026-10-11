import pytest
from fastapi.testclient import TestClient

from legal_rag_chatbot import api
from legal_rag_chatbot.llm.chat import Answer, Source


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    answer = Answer(
        text="Die Frist endet am 29.01.2026 [1].",
        sources=[
            Source(1, "CASE-023-07", "gs://bucket/a", "Klagefrist ..."),
            Source(2, None, "gs://bucket/b", "nicht zitiert"),
        ],
    )
    monkeypatch.setattr(api, "answer_with_sources", lambda question: answer)
    monkeypatch.setattr(api, "document_titles", lambda: {"CASE-023-07": "Klageschrift"})
    return TestClient(api.app)


def test_chat_returns_answer_and_cited_sources(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        json={"question": "Wann endet die Frist?", "document_ids": [], "history": []},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Die Frist endet")
    assert body["sources"] == [
        {
            "documentId": "CASE-023-07",
            "documentName": "CASE-023-07 – Klageschrift",
            "excerpt": "[1] Klagefrist ...",
        }
    ]


def test_chat_rejects_empty_question(client: TestClient) -> None:
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
