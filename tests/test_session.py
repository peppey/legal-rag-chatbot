from legal_rag_chatbot.rag.session import SessionStore

OWNER = "user-1"


def test_search_returns_best_chunk_of_selected_documents(embedder) -> None:
    store = SessionStore(embedder=embedder)
    rent = store.add(
        "miete.txt",
        b"Die Miete betraegt 900 Euro. Miete ist monatlich faellig.",
        owner=OWNER,
    )
    notice = store.add(
        "kuendigung.txt",
        b"Die Kuendigung erfolgt schriftlich. Kuendigung per Post.",
        owner=OWNER,
    )
    store.add("urheber.txt", b"Urheberrecht des Fotografen.", owner=OWNER)

    hits = store.search(
        "Wie erfolgt die Kuendigung?", [rent.id, notice.id], top_k=2, owner=OWNER
    )

    assert [h.document_id for h in hits] == [notice.id, rent.id]
    assert hits[0].document_name == "kuendigung.txt"
    assert hits[0].page is None


def test_search_ignores_unknown_and_removed_documents(embedder) -> None:
    store = SessionStore(embedder=embedder)
    document = store.add("miete.txt", b"Die Miete betraegt 900 Euro.", owner=OWNER)

    assert store.search("Miete", ["unbekannt"], top_k=3, owner=OWNER) == []
    assert store.remove(document.id, OWNER)
    assert store.search("Miete", [document.id], top_k=3, owner=OWNER) == []
    assert not store.remove(document.id, OWNER)


def test_full_text_returns_all_chunks_only_within_word_limit(embedder) -> None:
    store = SessionStore(embedder=embedder)
    document = store.add("miete.txt", b"Die Miete betraegt 900 Euro.", owner=OWNER)

    hits = store.full_text([document.id], max_words=100, owner=OWNER)
    assert [h.text for h in hits] == ["Die Miete betraegt 900 Euro."]
    assert store.full_text([document.id], max_words=3, owner=OWNER) == []


def test_uploads_are_invisible_to_other_users(embedder) -> None:
    store = SessionStore(embedder=embedder)
    document = store.add("miete.txt", b"Die Miete betraegt 900 Euro.", owner=OWNER)

    assert store.get(document.id, "user-2") is None
    assert store.search("Miete", [document.id], top_k=3, owner="user-2") == []
    assert store.full_text([document.id], 100, owner="user-2") == []
    assert not store.remove(document.id, "user-2")
    assert store.get(document.id, OWNER) is document
