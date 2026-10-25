"""Prompt injection evaluation: python -m legal_rag_chatbot.evaluation.injection_eval

The search is replaced by fixed excerpts, one of which carries an injection. The answers
are checked with regular expressions, so no judge model is needed.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from legal_rag_chatbot import prompts
from legal_rag_chatbot.access import SYSTEM_ACCESS
from legal_rag_chatbot.evaluation.answer_eval import _with_retry
from legal_rag_chatbot.evaluation.dataset import EVAL_PATH, new_run_id, report_path
from legal_rag_chatbot.llm import chat
from legal_rag_chatbot.llm.chat import MODEL, Turn

INJECTION_EVAL_PATH = EVAL_PATH.with_name("injection_eval.json")


def load_cases(path: Path = INJECTION_EVAL_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)["cases"]


def check(case: dict[str, Any], answer: str) -> list[str]:
    """Reasons why the answer fails the case, empty if it passes."""
    problems = [
        f"verbotenes Muster: {pattern}"
        for pattern in case.get("fail_if", [])
        if re.search(pattern, answer, re.MULTILINE)
    ]
    problems += [
        f"fehlt: {text}" for text in case.get("must_contain", []) if text not in answer
    ]
    return problems


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    chunks = [SimpleNamespace(text=t, source_uri="gs://eval/injection") for t in case["excerpts"]]
    history = [Turn(**turn) for turn in case.get("history", [])]
    with patch.object(chat, "fetch_contexts", lambda *args, **kwargs: chunks):
        answer = _with_retry(
            lambda: chat.answer_question(
                case["question"], history=history, access=SYSTEM_ACCESS
            )
        )
    problems = check(case, answer)
    return {
        "id": case["id"],
        "kind": case["kind"],
        "passed": not problems,
        "problems": problems,
        "answer": answer,
    }


def main() -> None:
    """Run all cases, print the result per case and save the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Where to write the JSON report")
    args = parser.parse_args()
    run_id = new_run_id()

    results = [run_case(case) for case in load_cases()]
    for r in results:
        print(f"{r['id']} {r['kind']:28} {'OK' if r['passed'] else 'FEHLER'}")
        for problem in r["problems"]:
            print(f"    {problem}")
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} Fälle bestanden")

    output = args.output or report_path("injection", run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {"run_id": run_id, "model": MODEL, **prompts.info()}
    output.write_text(
        json.dumps(
            {"config": config, "passed": passed, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Bericht: {output}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
