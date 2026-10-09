import math


def dedupe(ranked: list[str]) -> list[str]:
    """Collapse multiple chunks of the same document, keeping the best rank."""
    return list(dict.fromkeys(ranked))


def recall_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    hits = sum(1 for doc in ranked[:k] if relevance.get(doc, 0) > 0)
    return hits / len(relevance)


def precision_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    hits = sum(1 for doc in ranked[:k] if relevance.get(doc, 0) > 0)
    return hits / k


def hit_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    return float(any(relevance.get(doc, 0) > 0 for doc in ranked[:k]))


def core_recall_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    """Recall over the core documents (relevance 3) only."""
    core = {doc for doc, rel in relevance.items() if rel == 3}
    if not core:
        return float("nan")
    return len(core & set(ranked[:k])) / len(core)


def reciprocal_rank(ranked: list[str], relevance: dict[str, int]) -> float:
    for rank, doc in enumerate(ranked, start=1):
        if relevance.get(doc, 0) > 0:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    def dcg(gains: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))

    actual = dcg([relevance.get(doc, 0) for doc in ranked[:k]])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return actual / ideal
