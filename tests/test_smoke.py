import importlib

import pytest

import legal_rag_chatbot


def test_main_prints_greeting(capsys: pytest.CaptureFixture[str]) -> None:
    legal_rag_chatbot.main()

    assert "legal-rag-chatbot" in capsys.readouterr().out


@pytest.mark.parametrize(
    "module",
    [
        "legal_rag_chatbot.ingestion.upload",
        "legal_rag_chatbot.llm.chat",
        "legal_rag_chatbot.rag.corpus",
        "legal_rag_chatbot.rag.query",
        "legal_rag_chatbot.scripts.upload_dataset",
        "legal_rag_chatbot.scripts.evaluate_retrieval",
        "legal_rag_chatbot.scripts.evaluate_answers",
    ],
)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
