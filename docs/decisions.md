# Decisions

As-built deviations from [`design.md`](design.md), and the reasoning behind the
stack choices. The design doc is the original spec and is left unedited; this
file records where reality diverged from it and why.

## pgvector, not TurboVec

**Supersedes "TurboVec, not Chroma" below.** The index is a `vector` column in
the same Postgres that already backs ingestion, queried with an exact `<=>`
scan. See [`rag/vectorstore.py`](../src/riftbound_bot/rag/vectorstore.py).

What forced it was an operational failure, not a retrieval one. The index lived
in a bind-mounted `./data`, and when that host directory went missing Docker
recreated it empty and root-owned — breaking bootstrap, and the index write
after it. Storing the index where the rest of the data already lived removed the
directory, the mount, and the ownership workaround the mount required.

The earlier decision's "no separate vector service to run" argument had already
gone moot: Postgres was a required service either way, so the embedded index
wasn't avoiding a container, only splitting the data across two places.

Not `langchain-postgres`: [`rag/chain.py`](../src/riftbound_bot/rag/chain.py)
makes exactly three calls into the store, so a `PGVector` adapter — and the
SQLAlchemy stack under it — would re-derive an abstraction this project doesn't
use, where [`ingest/db.py`](../src/riftbound_bot/ingest/db.py) had already
established raw psycopg3 as the idiom. That surface is now the `RetrievalStore`
Protocol.

No ANN index: at ~1,300 vectors an exact scan is sub-millisecond and gives
strictly better recall than the 4-bit quantization it replaced. Revisit north of
~100k rows.

Two consequences worth knowing:

- **The bot now requires Postgres at runtime**, reversing the split where it
  only ever read a local file.
- **Postgres holds the only live copy of the hand-translated rules.** The
  corpus shipped in [`ingest/seeds/`](../src/riftbound_bot/ingest/seeds/) seeds
  an empty database and is never re-applied over existing rows, so edits made in
  the database need `rules_export` or `pg_dump` to survive.

Relevance scores keep TurboVec's `(cosine + 1) / 2` mapping so
`RETRIEVAL_SCORE_THRESHOLD` doesn't silently change meaning — see
`_relevance` in `rag/vectorstore.py`.

## TurboVec, not Chroma

**Superseded by "pgvector, not TurboVec" above; kept for the reasoning.**

An embedded, quantized vector index rather than a full vector database server —
no separate container to run, keeping the same "local file, no network service"
shape Chroma had.

`bit_width=4` prioritizes recall over compression: TurboVec's compression story
targets millions of vectors, which is irrelevant at this corpus's ~1,300-vector
scale. Cosine similarity is `TurboQuantVectorStore`'s unconditional behavior in
the pinned release (0.8.0) — there's no `similarity=` kwarg.

## Postgres, not MongoDB

The original plan called for MongoDB. MongoDB 5.0+ requires a CPU with AVX
support, and this project's target deployment hardware (an Intel Celeron J4105 /
Goldmont Plus) has none — `mongod` crashes on startup there.

Postgres has no such constraint and gets the same flexible, per-record-upsert
JSONB storage via one narrow-plus-JSONB table per corpus. See
[`ingest/db.py`](../src/riftbound_bot/ingest/db.py).

## DeepSeek by default, not Claude

**This deviates from the design doc.** The doc picked Claude specifically over
DeepSeek for grounding and hallucination reasons — DeepSeek's higher
hallucination rate is exactly what a strict-grounding rules-adjudication tool
can least afford.

The current default (`deepseek-v4-flash-free` via the free `opencode.ai/zen`
gateway) trades that quality margin for $0 cost.

Switching back is a config change, not a rewrite. The same gateway also serves
Claude and other frontier models under that base URL (paid, unlike the DeepSeek
default):

```bash
GENERATION_MODEL=claude-sonnet-5
```

Any other OpenAI-API-compatible provider works the same way by pointing
`GENERATION_API_BASE_URL` at it instead. `GENERATION_API_KEY` can stay blank for
the free default.
