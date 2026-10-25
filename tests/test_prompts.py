import pytest

from legal_rag_chatbot import prompts

PROMPT_NAMES = [
    "system",
    "format",
    "citation",
    "security",
    "upload",
    "rewrite",
    "route",
    "smalltalk",
    "judge",
    "judge_answerable",
    "judge_unanswerable",
]


def test_default_version_has_all_prompts():
    assert prompts.DEFAULT_VERSION in prompts.available_versions()
    for name in PROMPT_NAMES:
        assert prompts.get(name, prompts.DEFAULT_VERSION)


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        prompts.get("system", "v999")


def test_fingerprint_changes_with_content(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)
    version = tmp_path / "v1"
    version.mkdir()
    (version / "system.md").write_text("Hallo", encoding="utf-8")
    first = prompts.fingerprint("v1")
    assert first == prompts.fingerprint("v1")
    (version / "system.md").write_text("Hallo Welt", encoding="utf-8")
    assert prompts.fingerprint("v1") != first


def test_chat_prompts_are_built_from_files():
    from legal_rag_chatbot.llm import chat

    assert chat.SYSTEM_PROMPT.startswith("\nDu bist ein juristischer")
    assert "Antwortformat" in chat.SYSTEM_PROMPT
    assert "Dokumentauszüge bereitgestellt" in chat.RERANK_SYSTEM_PROMPT
    assert chat.RERANK_SYSTEM_PROMPT.endswith(chat.CITATION_INSTRUCTION)


def test_info_names_version_and_hash():
    info = prompts.info()
    assert info["prompt_version"] == prompts.VERSION
    assert len(info["prompt_hash"]) == 12
