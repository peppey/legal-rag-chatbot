from legal_rag_chatbot.rag.session import SessionStore


def test_search_returns_best_chunk_of_selected_documents(embedder) -> None:
    store = SessionStore(embedder=embedder)
    rent = store.add(
        "miete.txt", b"Die Miete betraegt 900 Euro. Miete ist monatlich faellig."
    )
    notice = store.add(
        "kuendigung.txt", b"Die Kuendigung erfolgt schriftlich. Kuendigung per Post."
    )
    store.add("urheber.txt", b"Urheberrecht des Fotografen.")

    hits = store.search("Wie erfolgt die Kuendigung?", [rent.id, notice.id], top_k=2)

    assert [h.document_id for h in hits] == [notice.id, rent.id]
    assert hits[0].document_name == "kuendigung.txt"
    assert hits[0].page is None


def test_search_ignores_unknown_and_removed_documents(embedder) -> None:
    store = SessionStore(embedder=embedder)
    document = store.add("miete.txt", b"Die Miete betraegt 900 Euro.")

    assert store.search("Miete", ["unbekannt"], top_k=3) == []
    assert store.remove(document.id)
    assert store.search("Miete", [document.id], top_k=3) == []
    assert not store.remove(document.id)
