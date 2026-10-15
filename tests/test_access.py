import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from legal_rag_chatbot import api
from legal_rag_chatbot.access import (
    GROUPS,
    SYSTEM_ACCESS,
    UnknownRole,
    access_for_roles,
    chunk_file_name,
    document_groups,
    load_policy,
)
from legal_rag_chatbot.rag import query
from legal_rag_chatbot.rag.session import SessionStore


def test_roles_map_to_their_groups() -> None:
    assert access_for_roles("u", ["a"]).groups == {"A", "Geteilt"}
    assert access_for_roles("u", ["b"]).groups == {"B", "Geteilt"}
    assert access_for_roles("u", ["admin"]).groups == set(GROUPS)
    assert access_for_roles("u", ["a", "b"]).groups == set(GROUPS)
    assert SYSTEM_ACCESS.groups == set(GROUPS)


def test_unknown_roles_grant_nothing() -> None:
    for roles in ([], [""], ["root"]):
        with pytest.raises(UnknownRole):
            access_for_roles("u", roles)


def test_policy_only_names_known_groups() -> None:
    assert {g for groups in load_policy().values() for g in groups} <= set(GROUPS)


def _auth_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("AUTH_MODE", "dev")
    ocr_dir, pdf_dir = tmp_path / "ocr", tmp_path / "pdf"
    ocr_dir.mkdir()
    for doc_id, group in [
        ("CASE-001-01", "A"),
        ("CASE-002-01", "B"),
        ("X-1", "Geteilt"),
    ]:
        (ocr_dir / f"{doc_id}.json").write_text(
            json.dumps({"document_id": doc_id, "access_group": group})
        )
        (pdf_dir / group).mkdir(parents=True, exist_ok=True)
        (pdf_dir / group / f"{doc_id}.pdf").write_bytes(b"%PDF " + doc_id.encode())
    monkeypatch.setattr(api, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(api, "document_groups", lambda: document_groups(ocr_dir))
    return TestClient(api.app)


def test_requests_without_dev_mode_or_with_unknown_role_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(api.app)
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.setenv("DEV_ROLE", "admin")
    assert client.post("/api/chat", json={"question": "x"}).status_code == 401
    assert client.get("/api/documents/CASE-001-01/file").status_code == 401

    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("DEV_ROLE", "root")
    assert client.post("/api/chat", json={"question": "x"}).status_code == 403
    monkeypatch.delenv("DEV_ROLE")
    assert client.post("/api/chat", json={"question": "x"}).status_code == 403


@pytest.mark.parametrize(
    ("role", "readable"),
    [
        ("a", {"CASE-001-01", "X-1"}),
        ("b", {"CASE-002-01", "X-1"}),
        ("admin", {"CASE-001-01", "CASE-002-01", "X-1"}),
    ],
)
def test_pdfs_are_served_only_within_the_groups_of_the_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str, readable: set[str]
) -> None:
    client = _auth_client(monkeypatch, tmp_path)
    headers = {"X-Dev-Role": role}

    for doc_id in ("CASE-001-01", "CASE-002-01", "X-1"):
        response = client.get(f"/api/documents/{doc_id}/file", headers=headers)
        assert response.status_code == (200 if doc_id in readable else 404)
        if doc_id in readable:
            assert response.content == b"%PDF " + doc_id.encode()


def test_unknown_document_ids_and_traversal_are_not_served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _auth_client(monkeypatch, tmp_path)
    headers = {"X-Dev-Role": "admin"}
    assert client.get("/api/documents/unknown/file", headers=headers).status_code == 404
    assert (
        client.get("/api/documents/..%2Fsecret/file", headers=headers).status_code
        == 404
    )


def test_the_role_is_passed_on_to_the_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = {}

    def fake_answer(question, **kwargs):
        received.update(kwargs)
        return api.Answer(text="ok")

    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setattr(api, "answer_with_sources", fake_answer)

    TestClient(api.app).post(
        "/api/chat", json={"question": "x"}, headers={"X-Dev-Role": "a"}
    )

    assert received["access"].groups == {"A", "Geteilt"}


def test_uploads_of_one_user_are_not_served_to_another(
    monkeypatch: pytest.MonkeyPatch, embedder
) -> None:
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setattr(api.session, "store", SessionStore(embedder=embedder))
    monkeypatch.setattr(api, "document_groups", lambda: {})
    client = TestClient(api.app)
    mine = {"X-Dev-Role": "a"}
    theirs = {"X-Dev-Role": "b"}

    upload = client.post(
        "/api/documents/upload", files={"file": ("n.txt", b"Geheim")}, headers=mine
    ).json()

    url = f"/api/documents/{upload['id']}/file"
    assert client.get(url, headers=mine).status_code == 200
    assert client.get(url, headers=theirs).status_code == 404
    assert (
        client.delete(f"/api/documents/{upload['id']}", headers=theirs).status_code
        == 204
    )
    assert client.get(url, headers=mine).status_code == 200


def _fake_rag(monkeypatch: pytest.MonkeyPatch, files: dict[str, str]):
    """Fake corpus whose RagFiles are display name to id; records what retrieval asks for."""
    requests = []

    def retrieve_contexts(*, vertex_rag_store, query):
        requests.append(vertex_rag_store.rag_resources[0].rag_file_ids)
        return SimpleNamespace(contexts=SimpleNamespace(contexts=["hit"]))

    client = SimpleNamespace(rag=SimpleNamespace(retrieve_contexts=retrieve_contexts))
    index = {chunk_file_name(group): file_id for group, file_id in files.items()}
    query._rag_file_index.cache_clear()
    monkeypatch.setattr(query, "_rag_file_index", lambda: index)
    return client, requests


def test_retrieval_is_limited_to_the_files_of_the_allowed_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, requests = _fake_rag(monkeypatch, {"A": "1", "B": "2", "Geteilt": "3"})

    query.fetch_contexts("x", client=client, groups={"A", "Geteilt"})
    query.fetch_contexts("x", client=client, groups={"B"})

    assert requests == [["1", "3"], ["2"]]


def test_retrieval_never_searches_the_whole_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, requests = _fake_rag(monkeypatch, {"A": "1"})

    assert query.fetch_contexts("x", client=client, groups=set()) == []
    assert query.fetch_contexts("x", client=client, groups={"B"}) == []
    assert requests == []
