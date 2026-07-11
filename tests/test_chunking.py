"""Tests for the sentence-aware chunker in rag_core."""
import rag_core


def test_empty_text_yields_no_chunks():
    assert rag_core.chunk_text("") == []
    assert rag_core.chunk_text("   \n  \n") == []


def test_short_text_is_a_single_chunk():
    text = "This is one short sentence. And a second one."
    chunks = rag_core.chunk_text(text, size=512)
    assert len(chunks) == 1
    assert "short sentence" in chunks[0]


def test_long_text_splits_into_multiple_chunks():
    # 20 sentences of ~40 chars each, well over a 100-char chunk size
    text = " ".join(f"This is sentence number {i} here." for i in range(20))
    chunks = rag_core.chunk_text(text, size=100)
    assert len(chunks) > 1
    # no chunk should be dramatically larger than the target size
    assert all(len(c) <= 200 for c in chunks)


def test_sentences_are_not_split_midway():
    # each chunk should end on sentence punctuation (never a partial sentence)
    text = " ".join(f"Sentence {i} content goes here." for i in range(15))
    chunks = rag_core.chunk_text(text, size=80)
    for c in chunks:
        assert c.rstrip().endswith((".", "!", "?"))


def test_oversized_single_sentence_is_hard_split_on_words():
    # one sentence longer than the chunk size must still be broken up
    long_sentence = "word " * 100  # 500 chars, no sentence enders
    chunks = rag_core.chunk_text(long_sentence.strip() + ".", size=50)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)


def test_sentence_overlap_is_carried_forward():
    # with 1-sentence overlap, the last sentence of one chunk should reappear
    text = " ".join(f"Alpha{i} bravo charlie delta echo foxtrot." for i in range(10))
    chunks = rag_core.chunk_text(text, size=60)
    assert len(chunks) >= 2
    # some content continuity should exist between consecutive chunks
    assert any(
        chunks[i].split()[-1] in chunks[i + 1] for i in range(len(chunks) - 1)
    )


def test_paragraph_breaks_are_respected():
    text = "First paragraph sentence.\n\nSecond paragraph sentence."
    chunks = rag_core.chunk_text(text, size=512)
    joined = " ".join(chunks)
    assert "First paragraph" in joined
    assert "Second paragraph" in joined
