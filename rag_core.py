"""
Core logic for the Adaptive RAG app: vector database (DuckDB + VSS/HNSW),
chunking, embeddings (Ollama), retrieval, and the four adaptive functions
(judge_query, rewrite_query, retrieve, judge_retrieved).

This module is deliberately UI-free so it can be tested on its own;
app.py (the marimo app) imports from here.
"""
from __future__ import annotations

import os
import re
import json
from typing import Callable, Iterable

import duckdb
import ollama

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DB_PATH = os.environ.get("RAG_DB_PATH", "rag.duckdb")
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("RAG_EMBED_DIM", "768"))  # nomic-embed-text = 768

# Dataset (subset kept well under 1GB by limiting the number of articles)
HF_DATASET = os.environ.get("RAG_DATASET", "wikimedia/wikipedia")
HF_CONFIG = os.environ.get("RAG_DATASET_CONFIG", "20231101.simple")
N_ARTICLES = int(os.environ.get("RAG_N_ARTICLES", "2000"))

# Chunking (length based, ~CHUNK_SIZE characters, never splitting mid-sentence)
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP_SENTENCES = 1  # carry this many trailing sentences into the next chunk


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
def get_db(path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the VSS (vector similarity) extension loaded."""
    con = duckdb.connect(path)
    con.execute("INSTALL vss;")
    con.execute("LOAD vss;")
    # allow the HNSW index to be persisted inside a file-backed database
    con.execute("SET hnsw_enable_experimental_persistence = true;")
    return con


def _table_is_ready(con: duckdb.DuckDBPyConnection) -> bool:
    """True if the chunks table exists and already has rows."""
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'chunks'"
    ).fetchone()[0]
    if not exists:
        return False
    return con.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0


# --------------------------------------------------------------------------- #
# Chunking  (length based, sentence-boundary aware)
# --------------------------------------------------------------------------- #
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    # split on paragraph breaks first, then sentence enders, to avoid cutting
    # in the middle of a sentence or paragraph.
    sentences: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_RE.split(para) if s.strip())
    return sentences


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """
    Group sentences into chunks of roughly `size` characters without splitting
    inside a sentence. A long sentence that exceeds `size` is hard-split on
    word boundaries as a fallback.
    """
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        # fallback: a single sentence longer than the chunk size -> split on words
        if len(sent) > size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            words, buf, buf_len = sent.split(), [], 0
            for w in words:
                if buf_len + len(w) + 1 > size and buf:
                    chunks.append(" ".join(buf))
                    buf, buf_len = [], 0
                buf.append(w)
                buf_len += len(w) + 1
            if buf:
                chunks.append(" ".join(buf))
            continue

        if current_len + len(sent) + 1 > size and current:
            chunks.append(" ".join(current))
            # carry the last sentence into the next chunk for a little overlap
            current = current[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES else []
            current_len = sum(len(s) + 1 for s in current)
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------- #
# Embeddings (Ollama)
# --------------------------------------------------------------------------- #
def embed_one(text: str) -> list[float]:
    return ollama.embed(model=EMBED_MODEL, input=[text]).embeddings[0]


def embed_many(texts: list[str], batch: int = 64) -> list[list[float]]:
    """Batch-embed for speed (much faster than one call per text)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(ollama.embed(model=EMBED_MODEL, input=texts[i : i + batch]).embeddings)
    return out


# --------------------------------------------------------------------------- #
# First-run setup: download -> chunk -> embed -> insert -> HNSW index
# --------------------------------------------------------------------------- #
def setup_database(
    con: duckdb.DuckDBPyConnection,
    progress: Callable[[int, int, str], None] | None = None,
    n_articles: int = N_ARTICLES,
) -> None:
    """
    Idempotently build the vector store. If the chunks table is already
    populated this returns immediately. `progress(done, total, label)` is an
    optional callback so the UI can show a progress bar.
    """
    if _table_is_ready(con):
        return

    from datasets import load_dataset

    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER,
            doc_title   VARCHAR,
            chunk_text  VARCHAR,
            embedding   FLOAT[{EMBED_DIM}]
        );
        """
    )

    # stream the dataset and only take the first n_articles (keeps size small)
    ds = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)

    next_id = 0
    seen = 0
    for article in ds:
        if seen >= n_articles:
            break
        title = article.get("title", "")
        text = article.get("text", "") or ""
        chunks = chunk_text(text)
        if chunks:
            embeddings = embed_many(chunks)  # batched for speed
            rows = [
                (next_id + i, title, ch, emb)
                for i, (ch, emb) in enumerate(zip(chunks, embeddings))
            ]
            con.executemany(
                "INSERT INTO chunks (id, doc_title, chunk_text, embedding) VALUES (?, ?, ?, ?)",
                rows,
            )
            next_id += len(chunks)
        seen += 1
        if progress:
            progress(seen, n_articles, f"Embedding article {seen}/{n_articles}: {title[:40]}")

    # HNSW index for fast cosine similarity search
    con.execute(
        "CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING HNSW (embedding) "
        "WITH (metric = 'cosine');"
    )
    if progress:
        progress(n_articles, n_articles, f"Indexed {len(rows)} chunks.")


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def retrieve(
    con: duckdb.DuckDBPyConnection,
    query: str,
    k: int = 5,
    return_text: bool = True,
):
    """
    Neural IR over the chunk store. Returns the k closest chunks.
    If return_text is True -> list of dicts {id, title, text, distance};
    otherwise -> list of ids (useful for post-filtering).
    """
    qvec = embed_one(query)
    sql = f"""
        SELECT id, doc_title, chunk_text,
               array_cosine_distance(embedding, ?::FLOAT[{EMBED_DIM}]) AS dist
        FROM chunks
        ORDER BY dist
        LIMIT ?
    """
    res = con.execute(sql, [qvec, k]).fetchall()
    if not return_text:
        return [r[0] for r in res]
    return [
        {"id": r[0], "title": r[1], "text": r[2], "distance": r[3]} for r in res
    ]


# --------------------------------------------------------------------------- #
# LLM helpers (Ollama chat)
# --------------------------------------------------------------------------- #
def _chat(system: str, user: str, temperature: float = 0.2) -> str:
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature},
    )
    return resp["message"]["content"].strip()


def _extract_json(text: str):
    """Best-effort extraction of the first JSON object/array in a string."""
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


# --- 1. judge_query: do we even need retrieval? -------------------------------
def judge_query(query: str) -> bool:
    """
    Decide whether answering the query needs document retrieval.
    Returns True if retrieval is needed, False if the model can answer directly.
    """
    system = (
        "You are a routing classifier for a retrieval-augmented assistant.\n"
        "Task: decide whether answering the user's message requires looking up "
        "external documents (facts, specific entities, dates, niche knowledge), "
        "or whether it can be answered directly (chit-chat, math, reasoning, "
        "generic writing).\n"
        "Output: respond with ONLY a JSON object {\"retrieve\": true|false}.\n"
        "Examples:\n"
        "User: Hello, how are you? -> {\"retrieve\": false}\n"
        "User: What is 17 * 23? -> {\"retrieve\": false}\n"
        "User: Who was the first president of Ghana? -> {\"retrieve\": true}\n"
        "User: Summarize the history of the Eiffel Tower. -> {\"retrieve\": true}"
    )
    out = _chat(system, f"User: {query}")
    parsed = _extract_json(out)
    if isinstance(parsed, dict) and "retrieve" in parsed:
        return bool(parsed["retrieve"])
    # fallback: if unsure, retrieve (safer for a RAG system)
    return True


# --- 2. rewrite_query: produce n better retrieval queries ---------------------
def rewrite_query(query: str, original_query: str, n: int = 3) -> list[str]:
    """
    Transform/decompose a query into n search-friendly queries. `original_query`
    is the user's first question; `query` is the current focus (they differ when
    we loop back from judge_retrieved with a refined query).
    """
    system = (
        "You rewrite a user's question into multiple focused search queries for a "
        "document retrieval system (Neural IR over an encyclopedia).\n"
        "Guidelines (Task, Instruction, Context, Output, Style = TICOS):\n"
        "- Task: produce diverse, specific queries that together cover what is "
        "needed to answer the question.\n"
        "- Keep each query short, keyword-rich, and standalone.\n"
        f"- Output ONLY a JSON array of exactly {n} strings.\n"
        "Example:\n"
        "Original: Tell me about the Eiffel Tower.\n"
        "Current: Tell me about the Eiffel Tower.\n"
        '-> ["Eiffel Tower history", "Eiffel Tower height and construction", '
        '"Eiffel Tower architect Gustave Eiffel"]'
    )
    user = f"Original: {original_query}\nCurrent: {query}\nReturn {n} queries."
    out = _chat(system, user, temperature=0.4)
    parsed = _extract_json(out)
    if isinstance(parsed, list) and parsed:
        queries = [str(q).strip() for q in parsed if str(q).strip()]
        if queries:
            return queries[:n]
    # fallback: just use the query itself
    return [query]


# --- 4. judge_retrieved: stop, or refine and loop -----------------------------
def judge_retrieved(
    retrieved: list[dict],
    original_query: str,
    n: int = 3,
):
    """
    Look at the retrieved chunks and decide:
      - return None  -> documents are sufficient, stop retrieving;
      - return list[str] -> not sufficient, here are new refined queries to
        feed back into the rewrite/retrieve loop.
    """
    context = "\n\n".join(
        f"[{i+1}] ({d['title']}) {d['text'][:300]}" for i, d in enumerate(retrieved)
    )
    system = (
        "You are the relevance judge in an adaptive RAG loop.\n"
        "Given the user's question and the retrieved passages, decide if they are "
        "sufficient to answer well.\n"
        "Output ONLY a JSON object:\n"
        '  {"sufficient": true}  if the passages are enough, OR\n'
        f'  {{"sufficient": false, "new_queries": [..{n} refined search queries..]}} '
        "if more/different retrieval is needed."
    )
    user = f"Question: {original_query}\n\nRetrieved passages:\n{context}"
    out = _chat(system, user)
    parsed = _extract_json(out)
    if isinstance(parsed, dict):
        if parsed.get("sufficient") is True:
            return None
        nq = parsed.get("new_queries")
        if isinstance(nq, list) and nq:
            return [str(q).strip() for q in nq if str(q).strip()][:n]
    # fallback: treat as sufficient to avoid infinite loops
    return None


# --- final answer generation --------------------------------------------------
def _build_answer_prompt(query: str, context_docs: list[dict] | None):
    if context_docs:
        context = "\n\n".join(
            f"[{i+1}] ({d['title']}) {d['text']}" for i, d in enumerate(context_docs)
        )
        system = (
            "You are a helpful assistant. Answer the user's question using the "
            "provided context passages. Cite sources as [n] where relevant. If the "
            "context does not contain the answer, say so honestly."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        system = "You are a helpful assistant. Answer the user's question directly."
        user = query
    return system, user


def generate_response(query: str, context_docs: list[dict] | None = None) -> str:
    """Generate the final answer, optionally grounded in retrieved context."""
    system, user = _build_answer_prompt(query, context_docs)
    return _chat(system, user, temperature=0.3)


def generate_response_stream(query: str, context_docs: list[dict] | None = None):
    """Like generate_response but yields the answer token-by-token (for streaming)."""
    system, user = _build_answer_prompt(query, context_docs)
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.3},
        stream=True,
    )
    for part in stream:
        piece = part["message"]["content"]
        if piece:
            yield piece
