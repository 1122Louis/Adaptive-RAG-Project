# Adaptive RAG

An **Adaptive Retrieval-Augmented Generation** chat application that decides *how much*
retrieval each query actually needs — instead of blindly searching for every message.
The system routes queries, rewrites them into focused searches, retrieves from a vector
store, **judges the quality of its own results**, and loops to refine when the retrieved
context is insufficient.

Built entirely with **local, open-source components** — no cloud APIs, no keys:
[marimo](https://marimo.io) (reactive UI), [DuckDB](https://duckdb.org) with the
VSS/HNSW extension (vector search), and [Ollama](https://ollama.com) (local LLM +
embeddings).

---

## Table of contents

- [Key idea](#key-idea)
- [Architecture](#architecture)
- [The adaptive pipeline](#the-adaptive-pipeline)
- [System design](#system-design)
  - [Component responsibilities](#component-responsibilities)
  - [Data flow: first run vs. query time](#data-flow-first-run-vs-query-time)
  - [Design decisions & trade-offs](#design-decisions--trade-offs)
- [Tech stack](#tech-stack)
- [Run it](#run-it)
- [Configuration](#configuration)
- [Project structure](#project-structure)

---

## Key idea

A naive RAG system retrieves documents for **every** query — wasteful for greetings or
arithmetic, and prone to injecting irrelevant context. **Adaptive RAG** adds two decision
points driven by the LLM itself:

1. **Before retrieving** — *does this query even need documents?* (routing)
2. **After retrieving** — *are these results good enough, or should I search again?*
   (self-critique / refinement loop)

This turns a one-shot "search → answer" pipeline into a small **agentic loop** that
adapts its own effort to the difficulty of the question.

---

## Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │                marimo app (app.py)            │
                        │   chat window · Neural-IR search · progress   │
                        └───────────────┬──────────────────────────────┘
                                        │  imports (UI-free core)
                                        ▼
                        ┌──────────────────────────────────────────────┐
                        │                 rag_core.py                   │
                        │  chunking · embeddings · retrieval · 4 funcs  │
                        └───────┬───────────────────────────┬──────────┘
                                │                           │
                   embeddings & chat                 vector search
                                │                           │
                                ▼                           ▼
                   ┌────────────────────────┐   ┌────────────────────────┐
                   │        Ollama          │   │        DuckDB          │
                   │  qwen2.5:7b-instruct   │   │  chunks table +        │
                   │  nomic-embed-text (768)│   │  HNSW index (cosine)   │
                   └────────────────────────┘   └────────────────────────┘
```

**Two-layer separation of concerns:** `rag_core.py` is deliberately **UI-free** so the
retrieval logic can be imported, scripted, and tested on its own. `app.py` is a thin
reactive presentation layer over it.

### Query-time control flow

```
          user message
               │
               ▼
     ┌───────────────────┐   no    ┌──────────────────────────┐
     │   judge_query()   │────────▶│  answer directly (stream) │
     │ retrieval needed? │         └──────────────────────────┘
     └─────────┬─────────┘
               │ yes
               ▼
     ┌───────────────────┐   ◀──────────────────────────┐
     │  rewrite_query()  │   refined queries             │
     │  → N search terms │                               │
     └─────────┬─────────┘                               │
               ▼                                         │
     ┌───────────────────┐                               │
     │    retrieve()     │  top-k per query, dedup       │
     │  HNSW cosine × N  │                               │
     └─────────┬─────────┘                               │
               ▼                                         │
     ┌───────────────────┐  insufficient (new queries)   │
     │ judge_retrieved() │───────────────────────────────┘
     │ sufficient?       │        (loop, ≤ MAX_ROUNDS)
     └─────────┬─────────┘
               │ sufficient
               ▼
     ┌───────────────────┐
     │ generate_response │  grounded answer, streamed token-by-token
     │ _stream()         │
     └───────────────────┘
```

---

## The adaptive pipeline

The four rubric functions in `rag_core.py`, in the order the loop calls them:

| # | Function | Role | Returns |
|---|----------|------|---------|
| 1 | `judge_query(query)` | **Router.** LLM classifies whether the query needs external docs (facts/entities/dates) vs. can be answered directly (chit-chat/math/reasoning). | `bool` |
| 2 | `rewrite_query(query, original_query, n)` | **Query expansion.** Decomposes the question into `n` diverse, keyword-rich search queries (TICOS-prompted) to improve recall. | `list[str]` |
| 3 | `retrieve(con, query, k)` | **Neural IR.** Embeds the query and runs top-`k` cosine search over the HNSW-indexed chunk store. | `list[dict]` |
| 4 | `judge_retrieved(retrieved, original_query, n)` | **Self-critique.** LLM judges whether the retrieved passages are sufficient; if not, emits `n` refined queries to feed back into the loop. | `None` \| `list[str]` |

Each LLM decision is prompted to return **strict JSON**, parsed defensively by
`_extract_json()`, and every function has a **safe fallback** (e.g. "when unsure,
retrieve"; "treat as sufficient to avoid infinite loops") so a malformed model response
never breaks the loop.

---

## System design

### Component responsibilities

| Layer | File / component | Responsibility |
|-------|------------------|----------------|
| **Presentation** | `app.py` (marimo cells) | Chat UI, Neural-IR search table, first-run progress bar, config constants, streaming orchestration. |
| **Orchestration** | `adaptive_chat()` generator | Drives the router → rewrite → retrieve → judge → answer loop; emits live status as *reasoning* chunks and the answer as *message* chunks. |
| **Core logic** | `rag_core.py` | Chunking, embeddings, retrieval, the four adaptive functions, and answer generation — all UI-free. |
| **Vector store** | DuckDB + VSS | `chunks(id, doc_title, chunk_text, embedding FLOAT[768])` with a persisted HNSW index (`metric='cosine'`). |
| **Models** | Ollama | `qwen2.5:7b-instruct` for chat/judging, `nomic-embed-text` for 768-dim embeddings. |

### Data flow: first run vs. query time

**First run (one-time ingestion — `setup_database`):**

```
HF wikipedia (streaming) ─▶ take N articles ─▶ chunk_text() ─▶ embed_many() (batched)
        ─▶ INSERT into chunks ─▶ CREATE HNSW INDEX (cosine) ─▶ cached in rag.duckdb
```

The dataset is **streamed** (never fully downloaded) and capped at `N_ARTICLES` to keep
the corpus well under 1 GB. Embeddings are **batched** (64 at a time) for throughput.
Ingestion is **idempotent** — if the `chunks` table already has rows, setup returns
immediately, so only the *first* launch pays the cost; later runs start instantly.

**Query time (per message):** the control-flow diagram above — router, then a bounded
retrieval-refinement loop, then a grounded, streamed answer.

### Design decisions & trade-offs

- **UI/core split.** Keeping `rag_core.py` free of marimo makes the retrieval logic
  unit-testable and reusable outside the notebook.
- **DuckDB as the vector store.** A single embedded file (`rag.duckdb`) holds the data
  *and* the HNSW index — zero external services, trivial to ship and cache. HNSW
  persistence in a file-backed DB is still experimental, hence
  `SET hnsw_enable_experimental_persistence = true`.
- **HNSW + cosine.** Approximate nearest-neighbour search keeps retrieval fast as the
  corpus grows; cosine distance matches how `nomic-embed-text` embeddings are compared.
- **Sentence-aware chunking.** `chunk_text()` groups whole sentences up to ~512 chars
  (with 1-sentence overlap) and only hard-splits sentences longer than the chunk size —
  preserving semantic units so embeddings stay meaningful.
- **Multi-query retrieval + dedup.** Rewriting into `N` queries and merging results
  (de-duplicated by chunk `id`) raises recall beyond a single embedding lookup.
- **Bounded refinement loop.** `MAX_ROUNDS` caps how many times the judge can send the
  system back to retrieve, guaranteeing termination and predictable latency.
- **Defensive JSON + fallbacks.** Local 7B models are not always well-behaved; every
  decision point degrades gracefully rather than crashing.
- **Streaming UX.** marimo streams generators in "delta" mode, so status/"thinking" is
  emitted as **reasoning** chunks (shown live, not saved) while the **answer** is yielded
  as plain strings (the persisted conversation message).

---

## Tech stack

| Concern | Choice | Why |
|---------|--------|-----|
| UI / app runtime | **marimo** | Reactive Python notebook-as-app; built-in chat, tables, progress bars. |
| Vector store & search | **DuckDB + VSS/HNSW** | Embedded, file-based, fast ANN search — no separate DB server. |
| Chat model | **qwen2.5:7b-instruct** (via Ollama) | Capable ≤7B instruct model that runs locally. |
| Embeddings | **nomic-embed-text** (via Ollama) | 768-dim, strong retrieval quality, runs locally. |
| Corpus | **Simple-English Wikipedia** (HF) | Clean, general-knowledge text; streamed and capped for size. |

---

## Run it

```bash
uv run marimo edit app.py --sandbox      # editing / development
uv run marimo run  app.py --sandbox      # app view (deliverable)
```

Requires a running **Ollama** with the models pulled:

```bash
ollama pull qwen2.5:7b-instruct      # chat model (<= 7B)
ollama pull nomic-embed-text         # embedding model (768-dim)
```

On **first run**, the app downloads a subset of Simple-English Wikipedia, chunks +
embeds it, and builds the HNSW index (progress bar shown). The result is cached in
`rag.duckdb`, so later runs start instantly.

---

## Configuration

Edit the constants in the config cell of `app.py`:

| Constant | Meaning | Default |
|----------|---------|---------|
| `N_QUERIES` | Search queries the rewriter produces | 3 |
| `K` | Results retrieved per query | 5 |
| `MAX_ROUNDS` | Max refine loops in the retrieval judge | 2 |
| `N_ARTICLES` | Corpus size built on first run | 1000 |

Models, dataset, chunk size, and DB path can also be overridden via **environment
variables** (see the config block at the top of `rag_core.py`, e.g. `RAG_CHAT_MODEL`,
`RAG_EMBED_MODEL`, `RAG_N_ARTICLES`, `RAG_CHUNK_SIZE`).

---

## Project structure

```
.
├── app.py          # marimo app: chat window, Neural-IR search, first-run setup, streaming
├── rag_core.py     # UI-free core: DB/HNSW, chunking, embeddings, retrieval, 4 adaptive funcs
├── pyproject.toml  # dependencies (marimo, duckdb, datasets, ollama, huggingface-hub)
├── uv.lock         # pinned dependency lockfile
└── README.md
```

> `rag.duckdb` (the vector store) is generated on first run and git-ignored — it rebuilds
> automatically from the dataset.
