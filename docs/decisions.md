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

## `/card` reads the `cards` table, not the vector index

**This deviates from the design doc,** which scoped the bot to one command.
`/card` is a second one: exact card lookup by name, answered from stored data
with no LLM in the path.

It is also the first thing in the bot to read `cards` at request time — every
other runtime query goes to `embeddings`. Serving it from the index instead
would have been the smaller diff, and was rejected: embedding metadata
deliberately carries only what retrieval needs (id, both names, rarity, source
url), so energy/power/might/color/tags and the image would all have had to be
added to it, and widening that metadata means re-embedding the whole corpus for
data that has nothing to do with similarity search.

The whole table is loaded into memory once at startup. At ~1,256 rows that is
a fraction of a megabyte, and Discord's autocomplete contract requires it
anyway: a callback that fires per keystroke inside a 3-second budget cannot
afford a round trip per character. See
[`cards.py`](../src/riftbound_bot/cards.py).

## The bot refuses to start on card data with no images

Card rows written before the scraper captured `assets` have no `image_url`, and
`/card` exists to show a card face. Rendering the embed anyway would put a
card-shaped hole in a channel and look like a bug in Discord rather than stale
data, so `build_client` fails at boot with the command that fixes it.

That guard would otherwise turn this change into a bot that won't start, so
`bootstrap` backfills the field ahead of its index check — the one place it
refreshes data already present, and it stops doing work as soon as the data is
current. See `_ensure_card_images` in
[`ingest/bootstrap.py`](../src/riftbound_bot/ingest/bootstrap.py).
