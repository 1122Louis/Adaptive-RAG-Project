"""Tests for the best-effort JSON extractor used to parse LLM output."""
import rag_core


def test_extracts_plain_object():
    assert rag_core._extract_json('{"retrieve": true}') == {"retrieve": True}


def test_extracts_plain_array():
    assert rag_core._extract_json('["a", "b", "c"]') == ["a", "b", "c"]


def test_extracts_object_embedded_in_prose():
    text = 'Sure! Here is my decision: {"sufficient": false, "new_queries": ["x"]} done.'
    assert rag_core._extract_json(text) == {
        "sufficient": False,
        "new_queries": ["x"],
    }


def test_extracts_object_from_fenced_code_block():
    text = 'Answer:\n```json\n{"retrieve": false}\n```'
    assert rag_core._extract_json(text) == {"retrieve": False}


def test_returns_none_when_no_json_present():
    assert rag_core._extract_json("no json here at all") is None


def test_returns_none_on_malformed_json():
    assert rag_core._extract_json('{"retrieve": tru') is None
