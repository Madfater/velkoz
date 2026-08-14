# Decisions

As-built deviations from [`design.md`](design.md), and the reasoning behind the
stack choices. The design doc is the original spec and is left unedited; this
file records where reality diverged from it and why.

## TurboVec, not Chroma

An embedded, quantized vector index rather than a full vector database server —
no separate container to run, keeping the same "local file, no network service"
shape Chroma had.

`bit_width=4` prioritizes recall over compression: TurboVec's compression story
targets millions of vectors, which is irrelevant at this corpus's ~1,300-vector
scale. Cosine similarity is `TurboQuantVectorStore`'s unconditional behavior in
the pinned release (0.8.0) — there's no `similarity=` kwarg. A cosine/dot-product
toggle exists on turbovec's unreleased main branch, so re-check this if the pin is
ever bumped past 0.8.x. See [`rag/vectorstore.py`](../src/riftbound_bot/rag/vectorstore.py).

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
