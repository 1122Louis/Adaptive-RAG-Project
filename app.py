# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23",
#     "duckdb>=1.1",
#     "datasets",
#     "ollama",
# ]
# ///
#
# Adaptive RAG marimo app.
# Run with:   marimo edit app.py --sandbox      (or: marimo run app.py --sandbox)
#
# Streaming design (works on current marimo's "delta" streaming):
#   * Status / thinking updates are emitted as REASONING chunks
#     ({"type": "reasoning-..."}). They stream live to the UI but are NOT part of
#     the final conversation message.
#   * The final answer is yielded as plain strings, which marimo accumulates into
#     the message appended to the conversation (i.e. the "last yielded value").

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("# 🔎 Adaptive RAG &nbsp; — chat that decides how much retrieval it needs")
    return


@app.cell
def _():
    import rag_core

    # ---- configuration (edit these) ----
    N_QUERIES = 3      # how many search queries the rewriter produces (configurable)
    K = 5              # results retrieved per query
    MAX_ROUNDS = 2     # max refine loops in the retrieval judge
    N_ARTICLES = 1000  # corpus size built on first run (kept well under 1GB)
    return K, MAX_ROUNDS, N_ARTICLES, N_QUERIES, rag_core


@app.cell
def _(rag_core):
    con = rag_core.get_db()
    return (con,)


@app.cell
def _(N_ARTICLES, con, mo, rag_core):
    # First run: download data + build embeddings + HNSW index, with a progress bar.
    # Later runs: the table already exists, so this is instant.
    if rag_core._table_is_ready(con):
        _n = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
        setup_status = mo.md(f"✅ **Vector store ready** — {_n} chunks indexed (HNSW, cosine).")
    else:
        with mo.status.progress_bar(
            total=N_ARTICLES, title="Building vector store", subtitle="downloading + embedding"
        ) as _bar:
            rag_core.setup_database(
                con,
                progress=lambda d, t, label: _bar.update(increment=1, subtitle=label),
                n_articles=N_ARTICLES,
            )
        _n = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
        setup_status = mo.md(f"✅ **Built vector store** — {_n} chunks indexed (HNSW, cosine).")
    setup_status
    return


@app.cell
def _(K, MAX_ROUNDS, N_QUERIES, con, mo, rag_core):
    # The adaptive chat function (a generator). Status updates are emitted as
    # REASONING chunks (live "thinking", not part of the message). The final
    # answer is streamed as plain strings (the conversation message).
    def adaptive_chat(messages, config):
        rid = "status"  # id of the reasoning/thinking block

        def status(msg):
            return {"type": "reasoning-delta", "id": rid, "delta": msg + "\n\n"}

        user_query = messages[-1].content
        yield {"type": "reasoning-start", "id": rid}

        # 1. judge_query — do we even need retrieval?
        yield status("🤔 Judging whether retrieval is needed…")
        if not rag_core.judge_query(user_query):
            yield status("💬 No retrieval needed — answering directly.")
            yield {"type": "reasoning-end", "id": rid}
            yield from rag_core.generate_response_stream(user_query)
            return

        # 2-4. retrieval loop: rewrite -> retrieve -> judge_retrieved (refine?)
        original_query = user_query
        current_query = user_query
        docs = []
        for round_i in range(MAX_ROUNDS):
            yield status(f"✍️ Rewriting into {N_QUERIES} search queries (round {round_i + 1})…")
            queries = rag_core.rewrite_query(current_query, original_query, n=N_QUERIES)

            yield status("🔎 Retrieving for: " + "; ".join(queries))
            hits = []
            for q in queries:
                hits.extend(rag_core.retrieve(con, q, k=K))

            # de-duplicate retrieved chunks by id
            seen, docs = set(), []
            for d in hits:
                if d["id"] not in seen:
                    seen.add(d["id"])
                    docs.append(d)

            yield status(f"📚 Retrieved {len(docs)} unique chunks — judging sufficiency…")
            new_queries = rag_core.judge_retrieved(docs, original_query, n=N_QUERIES)
            if new_queries is None:
                yield status("✅ Context is sufficient.")
                break
            yield status("🔁 Not sufficient — refining queries and retrieving again.")
            current_query = " ".join(new_queries)

        # 5. final grounded answer, streamed as the conversation message
        yield status("📝 Generating grounded answer…")
        yield {"type": "reasoning-end", "id": rid}
        yield from rag_core.generate_response_stream(original_query, context_docs=docs[:K])

    chat = mo.ui.chat(adaptive_chat)
    chat
    return


@app.cell
def _(mo):
    mo.md("### 🗂️ Document search (Neural IR)")
    return


@app.cell
def _(mo):
    search_box = mo.ui.text(
        placeholder="Search the embedded documents…",
        full_width=True,
        label="Query",
    )
    search_box
    return (search_box,)


@app.cell
def _(con, mo, rag_core, search_box):
    if search_box.value.strip():
        _results = rag_core.retrieve(con, search_box.value, k=10)
        search_results = mo.ui.table(
            [
                {
                    "title": d["title"],
                    "distance": round(d["distance"], 4),
                    "text": d["text"],
                }
                for d in _results
            ],
            label=f"Top 10 results for: {search_box.value!r}",
        )
    else:
        search_results = mo.md("*Type a query above to search the embedded documents.*")
    search_results
    return


if __name__ == "__main__":
    app.run()
