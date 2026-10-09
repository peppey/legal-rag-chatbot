import math

from legal_rag_chatbot.evaluation import metrics
from legal_rag_chatbot.evaluation.retrieval_eval import map_chunk_to_document

RELEVANCE = {"A": 3, "B": 2, "C": 1}


def test_perfect_ranking() -> None:
    ranked = ["A", "B", "C", "X"]
    assert metrics.recall_at_k(ranked, RELEVANCE, 3) == 1.0
    assert metrics.ndcg_at_k(ranked, RELEVANCE, 3) == 1.0
    assert metrics.reciprocal_rank(ranked, RELEVANCE) == 1.0


def test_partial_ranking() -> None:
    ranked = ["X", "B", "Y", "A"]
    assert metrics.reciprocal_rank(ranked, RELEVANCE) == 0.5
    assert metrics.recall_at_k(ranked, RELEVANCE, 2) == 1 / 3
    assert metrics.precision_at_k(ranked, RELEVANCE, 4) == 0.5
    assert metrics.core_recall_at_k(ranked, RELEVANCE, 2) == 0.0
    assert metrics.hit_at_k(ranked, RELEVANCE, 1) == 0.0
    assert 0 < metrics.ndcg_at_k(ranked, RELEVANCE, 4) < 1


def test_no_relevant_retrieved() -> None:
    assert metrics.reciprocal_rank(["X"], RELEVANCE) == 0.0
    assert metrics.ndcg_at_k(["X"], RELEVANCE, 5) == 0.0
    assert math.isnan(metrics.core_recall_at_k(["A"], {"B": 2}, 5))


def test_dedupe_keeps_first_rank() -> None:
    assert metrics.dedupe(["A", "B", "A", "C"]) == ["A", "B", "C"]


def test_map_chunk_by_header_and_content() -> None:
    corpus = {"CASE-1-01": "erster text zum fall", "CASE-1-02": "zweiter langer text"}
    assert (
        map_chunk_to_document("document_id CASE-1-02\ncase_id CASE-1", corpus)
        == "CASE-1-02"
    )
    assert map_chunk_to_document("zweiter  langer\ntext", corpus) == "CASE-1-02"
    assert map_chunk_to_document("unbekannt", corpus) is None
