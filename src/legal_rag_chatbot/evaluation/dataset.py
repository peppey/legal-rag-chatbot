import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.json"
CORPUS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "synthetic_legal_counsel_documents.jsonl"
)
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"

NO_ANSWER_CATEGORY = "nicht_im_korpus"


def load_questions(path: Path = EVAL_PATH) -> list[dict[str, Any]]:
    """Load the evaluation questions from the dataset file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)["questions"]


def load_corpus(path: Path = CORPUS_PATH) -> dict[str, dict[str, Any]]:
    """Return the corpus documents keyed by document_id."""
    with path.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    return {doc["document_id"]: doc for doc in docs}
