import pytest
from fastapi.testclient import TestClient

from legal_rag_chatbot import api
from legal_rag_chatbot.llm.chat import Answer, Source
from legal_rag_chatbot.rag.session import SessionStore


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    answer = Answer(
        text="Die Frist endet am 29.01.2026 [1].",
        sources=[
            Source(1, "CASE-023-07", "gs://bucket/a", "Klagefrist ..."),
            Source(2, None, "gs://bucket/b", "nicht zitiert"),
        ],
    )
    monkeypatch.setattr(
        api, "answer_with_sources", lambda question, uploaded_document_ids=(): answer
    )
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


@pytest.fixture
def upload_client(monkeypatch: pytest.MonkeyPatch, embedder) -> TestClient:
    store = SessionStore(embedder=embedder)
    monkeypatch.setattr(api.session, "store", store)
    monkeypatch.setattr(api, "document_titles", lambda: {})
    return TestClient(api.app)


def test_upload_serve_and_delete(upload_client: TestClient) -> None:
    text = "Die Kuendigung muss bis zum 31.03.2026 erfolgen."
    response = upload_client.post(
        "/api/documents/upload", files={"file": ("notiz.txt", text.encode())}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "notiz.txt"
    assert body["size"] == len(text)

    file_url = f"/api/documents/{body['id']}/file"
    assert upload_client.get(file_url).content == text.encode()
    assert upload_client.delete(f"/api/documents/{body['id']}").status_code == 204
    assert upload_client.get(file_url).status_code == 404


def test_upload_rejects_unsupported_and_empty_files(upload_client: TestClient) -> None:
    for name, data in [("bild.png", b"x"), ("leer.txt", b"  \n")]:
        response = upload_client.post(
            "/api/documents/upload", files={"file": (name, data)}
        )
        assert response.status_code == 422
