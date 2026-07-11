"""
Tests for the four adaptive functions' parsing and fallback logic.

These mock the LLM call (`rag_core._chat`) so they run fast and offline —
we're testing the control logic around the model, not the model itself.
"""
import pytest

import rag_core


@pytest.fixture
def fake_chat(monkeypatch):
    """Replace the LLM call with a canned response string."""
    def _install(response: str):
        monkeypatch.setattr(rag_core, "_chat", lambda *a, **k: response)
    return _install


# --- judge_query -------------------------------------------------------------
def test_judge_query_true(fake_chat):
    fake_chat('{"retrieve": true}')
    assert rag_core.judge_query("Who was the first president of Ghana?") is True


def test_judge_query_false(fake_chat):
    fake_chat('{"retrieve": false}')
    assert rag_core.judge_query("Hello there") is False


def test_judge_query_falls_back_to_retrieve_on_garbage(fake_chat):
    # unparseable output -> safer default for a RAG system is to retrieve
    fake_chat("I cannot decide")
    assert rag_core.judge_query("anything") is True


# --- rewrite_query -----------------------------------------------------------
def test_rewrite_query_returns_parsed_list(fake_chat):
    fake_chat('["Eiffel Tower history", "Eiffel Tower height", "Gustave Eiffel"]')
    out = rag_core.rewrite_query("Tell me about the Eiffel Tower", "orig", n=3)
    assert out == ["Eiffel Tower history", "Eiffel Tower height", "Gustave Eiffel"]


def test_rewrite_query_caps_at_n(fake_chat):
    fake_chat('["a", "b", "c", "d", "e"]')
    out = rag_core.rewrite_query("q", "q", n=2)
    assert out == ["a", "b"]


def test_rewrite_query_falls_back_to_original_on_garbage(fake_chat):
    fake_chat("not a json array")
    out = rag_core.rewrite_query("my query", "my query", n=3)
    assert out == ["my query"]


# --- judge_retrieved ---------------------------------------------------------
def _docs():
    return [{"title": "T", "text": "some passage text"} for _ in range(3)]


def test_judge_retrieved_sufficient_returns_none(fake_chat):
    fake_chat('{"sufficient": true}')
    assert rag_core.judge_retrieved(_docs(), "question", n=3) is None


def test_judge_retrieved_insufficient_returns_new_queries(fake_chat):
    fake_chat('{"sufficient": false, "new_queries": ["refined a", "refined b"]}')
    out = rag_core.judge_retrieved(_docs(), "question", n=3)
    assert out == ["refined a", "refined b"]


def test_judge_retrieved_falls_back_to_none_on_garbage(fake_chat):
    # unparseable output -> treat as sufficient to guarantee loop termination
    fake_chat("hmm")
    assert rag_core.judge_retrieved(_docs(), "question", n=3) is None
