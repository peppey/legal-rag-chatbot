"""Answer evaluation with an LLM judge: python -m legal_rag_chatbot.evaluation.answer_eval"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from legal_rag_chatbot import prompts
from legal_rag_chatbot.access import SYSTEM_ACCESS
from legal_rag_chatbot.evaluation.dataset import (
    NO_ANSWER_CATEGORY,
    load_questions,
    new_run_id,
    report_path,
)
from legal_rag_chatbot.llm.chat import (
    MODEL,
    REWRITE_FOLLOW_UPS,
    THINKING_BUDGET,
    Turn,
    answer_question,
    answer_question_grounded,
    standalone_question,
)
from legal_rag_chatbot.rag.query import PROJECT_ID

T = TypeVar("T")

# Answer functions take the question and the earlier chat messages (empty for most questions).
AnswerFn = Callable[[str, Sequence[Turn]], str]
ANSWER_FUNCTIONS: dict[str, AnswerFn] = {
    "rerank": lambda question, history: answer_question(
        question, history=history, access=SYSTEM_ACCESS
    ),
    "grounded": lambda question, history: answer_question_grounded(
        standalone_question(question, history), access=SYSTEM_ACCESS
    ),
}

JUDGE_MODEL = "gemini-2.5-pro"
JUDGE_LOCATION = "europe-west4"  # gemini-2.5-pro is not available in europe-west3

JUDGE_PROMPT = f"{prompts.get('judge')}\n"
INSTRUCTION_ANSWERABLE = prompts.get("judge_answerable")
INSTRUCTION_UNANSWERABLE = prompts.get("judge_unanswerable")


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


def evaluate_question(
    client: genai.Client,
    question: dict[str, Any],
    answer_fn: AnswerFn = ANSWER_FUNCTIONS["rerank"],
) -> dict[str, Any]:
    """Generate an answer for one question and judge it."""
    history = [Turn(**turn) for turn in question.get("history", [])]
    answer = _with_retry(lambda: answer_fn(question["question"], history)) or ""
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
    parser.add_argument("--mode", choices=ANSWER_FUNCTIONS, default="rerank")
    parser.add_argument("--output", type=Path, help="Where to write the JSON report")
    parser.add_argument(
        "--run-id",
        help="Timestamp in the report name, pass the id of a retrieval run to pair them",
    )
    args = parser.parse_args()
    answer_fn = ANSWER_FUNCTIONS[args.mode]
    run_id = args.run_id or new_run_id()

    questions = load_questions()[: args.limit]
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=JUDGE_LOCATION)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(lambda q: evaluate_question(client, q, answer_fn), questions)
        )

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

    output = args.output or report_path("answers", run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": run_id,
        "mode": args.mode,
        "model": MODEL,
        "judge_model": JUDGE_MODEL,
        "thinking_budget": THINKING_BUDGET,
        "rewrite_follow_ups": REWRITE_FOLLOW_UPS,
        **prompts.info(),
        "questions": len(questions),
    }
    output.write_text(
        json.dumps(
            {"config": config, "summary": summary, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nBericht: {output}")


if __name__ == "__main__":
    main()
