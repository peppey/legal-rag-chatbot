"""Retrieval evaluation: python -m legal_rag_chatbot.evaluation.retrieval_eval"""

import argparse
import json
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import agentplatform

from legal_rag_chatbot.evaluation import metrics
from legal_rag_chatbot.evaluation.dataset import (
    RESULTS_DIR,
    load_corpus,
    load_questions,
)
from legal_rag_chatbot.rag.query import (
    LOCATION,
    PROJECT_ID,
    Ranker,
    fetch_contexts,
    map_chunk_to_document,
    normalize,
)

ALL_KS = (1, 3, 5, 10, 20)
DEFAULT_TOP_K = 10
MAX_ATTEMPTS = 6
CONTENT_PATTERN = re.compile(r"^content (.+)", re.MULTILINE | re.DOTALL)


def ks_for(top_k: int) -> tuple[int, ...]:
    """Cutoffs that fit into the number of retrieved chunks."""
    return tuple(k for k in ALL_KS if k <= top_k)


def is_stale(chunk_text: str, doc_id: str, normalized_corpus: dict[str, str]) -> bool:
    """True if the indexed chunk text does not occur in the local document."""
    match = CONTENT_PATTERN.search(chunk_text)
    if not match or doc_id not in normalized_corpus:
        return False
    return normalize(match.group(1))[:60] not in normalized_corpus[doc_id]


def evaluate_question(
    question: dict[str, Any],
    client: agentplatform.Client,
    normalized_corpus: dict[str, str],
    top_k: int,
    ranker: Ranker | None = None,
    ranker_model: str | None = None,
) -> dict[str, Any]:
    """Retrieve contexts for one question and compute its retrieval metrics."""
    ks = ks_for(top_k)
    for attempt in range(MAX_ATTEMPTS):
        try:
            contexts = fetch_contexts(
                question["question"],
                top_k=top_k,
                client=client,
                ranker=ranker,
                ranker_model=ranker_model,
            )
            # Empty responses are sometimes transient; retry before accepting them.
            if contexts or attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(2**attempt)
        except Exception:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            # The LLM reranker intermittently times out (503), so back off longer.
            time.sleep(5 * 2**attempt)

    chunk_docs: list[str | None] = []
    stale = 0
    for context in contexts:
        doc_id = map_chunk_to_document(context.text, normalized_corpus)
        chunk_docs.append(doc_id)
        if doc_id and is_stale(context.text, doc_id, normalized_corpus):
            stale += 1

    ranked = metrics.dedupe([doc for doc in chunk_docs if doc])
    relevance = {
        d["document_id"]: d["relevance"] for d in question["relevant_documents"]
    }

    result: dict[str, Any] = {
        "id": question["id"],
        "category": question["category"],
        "difficulty": question["difficulty"],
        "retrieved_chunks": chunk_docs,
        "retrieved_documents": ranked,
        "unmapped_chunks": chunk_docs.count(None),
        "stale_chunks": stale,
    }
    if relevance:
        result["mrr"] = metrics.reciprocal_rank(ranked, relevance)
        for k in ks:
            result[f"recall@{k}"] = metrics.recall_at_k(ranked, relevance, k)
            result[f"core_recall@{k}"] = metrics.core_recall_at_k(ranked, relevance, k)
            result[f"precision@{k}"] = metrics.precision_at_k(ranked, relevance, k)
            result[f"hit@{k}"] = metrics.hit_at_k(ranked, relevance, k)
            result[f"ndcg@{k}"] = metrics.ndcg_at_k(ranked, relevance, k)
    return result


def _mean(values: list[float]) -> float:
    """Mean that ignores NaN values."""
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def aggregate(results: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, float]:
    """Average the metrics over all questions that have relevant documents."""
    names = ["mrr"] + [
        f"{m}@{k}"
        for m in ("hit", "recall", "core_recall", "precision", "ndcg")
        for k in ks
    ]
    scored = [r for r in results if "mrr" in r]
    out = {name: _mean([r[name] for r in scored]) for name in names}
    out["n"] = len(scored)
    return out


def print_table(
    title: str, groups: dict[str, dict[str, float]], ks: tuple[int, ...]
) -> None:
    """Print aggregated metrics as a table, skipping empty groups."""
    last = max(ks)
    columns = [
        "n",
        "mrr",
        "hit@5",
        "recall@5",
        f"recall@{last}",
        "precision@5",
        "ndcg@5",
        f"ndcg@{last}",
    ]
    print(f"\n{title}")
    print(f"{'':28}" + "".join(f"{c:>13}" for c in columns))
    for name, values in groups.items():
        if values["n"] == 0:
            continue
        cells = "".join(
            f"{values[c]:>13d}" if c == "n" else f"{values[c]:>13.3f}" for c in columns
        )
        print(f"{name:28}{cells}")


def run(
    top_k: int,
    limit: int | None,
    workers: int,
    ranker: Ranker | None = None,
    ranker_model: str | None = None,
) -> dict[str, Any]:
    """Evaluate all questions in parallel and return the full report."""
    ks = ks_for(top_k)
    questions = load_questions()[:limit]
    corpus = load_corpus()
    normalized_corpus = {
        doc_id: normalize(doc["content"]) for doc_id, doc in corpus.items()
    }
    unknown = {
        d["document_id"]
        for q in questions
        for d in q["relevant_documents"]
        if d["document_id"] not in corpus
    }
    if unknown:
        raise ValueError(f"Evaluation references unknown documents: {sorted(unknown)}")

    client = agentplatform.Client(project=PROJECT_ID, location=LOCATION)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                lambda q: evaluate_question(
                    q, client, normalized_corpus, top_k, ranker, ranker_model
                ),
                questions,
            )
        )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)
        by_difficulty[r["difficulty"]].append(r)

    summary = {
        "overall": aggregate(results, ks),
        "by_category": {k: aggregate(v, ks) for k, v in sorted(by_category.items())},
        "by_difficulty": {
            k: aggregate(v, ks) for k, v in sorted(by_difficulty.items())
        },
    }
    return {
        "config": {
            "top_k": top_k,
            "ks": ks,
            "ranker": ranker,
            "ranker_model": ranker_model,
            "questions": len(questions),
        },
        "summary": summary,
        "indexed_corpus_stale_chunks": sum(r["stale_chunks"] for r in results),
        "unmapped_chunks": sum(r["unmapped_chunks"] for r in results),
        "results": results,
    }


def main() -> None:
    """Run the retrieval evaluation, print the tables and save the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks per query"
    )
    parser.add_argument("--ranker", choices=["llm", "service"], help="Reranker type")
    parser.add_argument("--ranker-model", help="Model name of the reranker")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N questions")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Where to write the JSON report")
    args = parser.parse_args()

    if args.ranker and not args.ranker_model:
        parser.error("--ranker requires --ranker-model")

    report = run(args.top_k, args.limit, args.workers, args.ranker, args.ranker_model)

    summary = report["summary"]
    ks = tuple(report["config"]["ks"])
    print_table("Gesamt (ohne nicht_im_korpus)", {"overall": summary["overall"]}, ks)
    print_table("Nach Kategorie", summary["by_category"], ks)
    print_table("Nach Schwierigkeit", summary["by_difficulty"], ks)

    total_chunks = sum(len(r["retrieved_chunks"]) for r in report["results"])
    if report["indexed_corpus_stale_chunks"]:
        print(
            f"\nWARNUNG: {report['indexed_corpus_stale_chunks']}/{total_chunks} Chunks "
            "stimmen inhaltlich nicht mit der lokalen Korpusdatei überein. Der "
            "Vertex-RAG-Corpus ist vermutlich veraltet und muss neu importiert werden."
        )
    if report["unmapped_chunks"]:
        print(
            f"\nWARNUNG: {report['unmapped_chunks']}/{total_chunks} Chunks konnten "
            "keinem document_id zugeordnet werden."
        )

    output = (
        args.output or RESULTS_DIR / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=list), encoding="utf-8"
    )
    print(f"\nBericht: {output}")


if __name__ == "__main__":
    main()
