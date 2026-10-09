"""Answer evaluation with an LLM judge: python -m legal_rag_chatbot.evaluation.answer_eval"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from legal_rag_chatbot.evaluation.dataset import (
    NO_ANSWER_CATEGORY,
    RESULTS_DIR,
    load_questions,
)
from legal_rag_chatbot.llm.chat import answer_question
from legal_rag_chatbot.rag.query import LOCATION, PROJECT_ID

T = TypeVar("T")

JUDGE_MODEL = "gemini-2.5-pro"

JUDGE_PROMPT = """\
Du bewertest die Antwort eines juristischen RAG-Assistenten anhand einer Referenzantwort.

Frage: {question}

Referenzantwort: {reference}

Antwort des Systems: {answer}

{instruction}
"""

INSTRUCTION_ANSWERABLE = """\
Bewerte "correctness":
- "richtig": alle wesentlichen Fakten der Referenzantwort sind enthalten und korrekt.
- "teilweise": nur ein Teil der wesentlichen Fakten ist enthalten oder korrekt.
- "falsch": die Antwort fehlt, widerspricht der Referenz oder verweigert die Antwort.
Setze "hallucination" auf true, wenn die Antwort konkrete Tatsachen behauptet, die der Referenzantwort widersprechen.
Setze "abstained" auf true, wenn das System sagt, dass die Dokumente keine ausreichende Grundlage liefern."""

INSTRUCTION_UNANSWERABLE = """\
Die Frage ist anhand des Korpus NICHT beantwortbar (siehe Referenzantwort).
Setze "abstained" auf true, wenn das System ausdrücklich sagt, dass die Dokumente keine ausreichende Grundlage liefern.
Setze "hallucination" auf true, wenn das System stattdessen konkrete Tatsachen als Antwort behauptet.
Setze "correctness" auf "richtig", wenn abstained true und hallucination false ist, sonst auf "falsch"."""


class Judgement(BaseModel):
    reasoning: str
    correctness: Literal["richtig", "teilweise", "falsch"]
    hallucination: bool
    abstained: bool


def judge(client: genai.Client, question: dict[str, Any], answer: str) -> Judgement:
    """Let the judge model grade an answer against the reference answer."""
    unanswerable = question["category"] == NO_ANSWER_CATEGORY
    prompt = JUDGE_PROMPT.format(
        question=question["question"],
        reference=question["reference_answer"],
        answer=answer or "(keine Antwort)",
        instruction=INSTRUCTION_UNANSWERABLE
        if unanswerable
        else INSTRUCTION_ANSWERABLE,
    )
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=Judgement,
        ),
    )
    return response.parsed


def _with_retry(call: Callable[[], T], attempts: int = 6) -> T:
    """Retry a call with exponential backoff (5s, 10s, 20s, ...) for quota errors."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


def evaluate_question(client: genai.Client, question: dict[str, Any]) -> dict[str, Any]:
    """Generate an answer for one question and judge it."""
    answer = _with_retry(lambda: answer_question(question["question"])) or ""
    judgement = _with_retry(lambda: judge(client, question, answer))

    return {
        "id": question["id"],
        "category": question["category"],
        "difficulty": question["difficulty"],
        "question": question["question"],
        "reference_answer": question["reference_answer"],
        "answer": answer,
        **judgement.model_dump(),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute verdict, hallucination and abstention rates."""
    n = len(results)
    return {
        "n": n,
        "richtig": sum(r["correctness"] == "richtig" for r in results) / n,
        "teilweise": sum(r["correctness"] == "teilweise" for r in results) / n,
        "falsch": sum(r["correctness"] == "falsch" for r in results) / n,
        "hallucination": sum(r["hallucination"] for r in results) / n,
        "abstained": sum(r["abstained"] for r in results) / n,
    }


def print_table(title: str, groups: dict[str, dict[str, float]]) -> None:
    """Print aggregated judgement rates as a table."""
    columns = ["n", "richtig", "teilweise", "falsch", "hallucination", "abstained"]
    print(f"\n{title}")
    print(f"{'':28}" + "".join(f"{c:>15}" for c in columns))
    for name, values in groups.items():
        cells = "".join(
            f"{values[c]:>15d}" if c == "n" else f"{values[c]:>15.2f}" for c in columns
        )
        print(f"{name:28}{cells}")


def main() -> None:
    """Run the answer evaluation, print the tables and save the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Only evaluate the first N questions")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Where to write the JSON report")
    args = parser.parse_args()

    questions = load_questions()[: args.limit]
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda q: evaluate_question(client, q), questions))

    answerable = [r for r in results if r["category"] != NO_ANSWER_CATEGORY]
    unanswerable = [r for r in results if r["category"] == NO_ANSWER_CATEGORY]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in answerable:
        by_category[r["category"]].append(r)

    summary: dict[str, Any] = {
        "answerable": aggregate(answerable) if answerable else None,
        "unanswerable": aggregate(unanswerable) if unanswerable else None,
        "by_category": {k: aggregate(v) for k, v in sorted(by_category.items())},
    }
    if summary["answerable"]:
        print_table("Beantwortbare Fragen", {"gesamt": summary["answerable"]})
        print_table("Nach Kategorie", summary["by_category"])
    if summary["unanswerable"]:
        print_table(
            "Nicht im Korpus (richtig = korrekt verweigert)",
            {"gesamt": summary["unanswerable"]},
        )

    output = args.output or RESULTS_DIR / f"answers_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"summary": summary, "results": results}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nBericht: {output}")


if __name__ == "__main__":
    main()
