# Data pipeline

```
data/rules/core_rules_zh_tw.md   # self-translated Traditional Chinese rules corpus (authoring source)
data/cards/cards.json            # historical snapshot only — no longer read by any code path
data/turbovec/                   # built vector index (gitignored, rebuild with build_index)
```

Postgres (`cards`/`rules` tables, JSONB-backed — see
[`ingest/db.py`](../src/riftbound_bot/ingest/db.py)) sits between the raw sources
above and the vector index: `cards_scrape.py`/`cards_from_api.py` upsert scraped
cards directly into the `cards` table (per-record, not a whole-file replace),
`rules_sync.py` parses `data/rules/*.md` and upserts into the `rules` table, and
`build_index.py` reads both tables to build the vector store.

This is all ingest-time. The live bot only ever reads the already-built
`data/turbovec/` index, never Postgres directly — see `load_vectorstore` in
[`rag/vectorstore.py`](../src/riftbound_bot/rag/vectorstore.py).

The sections below describe each step individually. You don't have to run them
by hand on a fresh deployment:
[`ingest/bootstrap.py`](../src/riftbound_bot/ingest/bootstrap.py) chains them in
order (rules sync → card scrape → build index) and is what `docker compose up`
runs before starting the bot. It only fills in what's missing — it skips the card
scrape when the `cards` table is already populated, and does nothing at all once
`data/turbovec/` exists — so refreshing data that's already there is still a
deliberate manual run of the steps below.

## Rules corpus

`data/rules/core_rules_zh_tw.md` currently holds a **sample slice** — the
Golden/Silver Rules, the start of deck construction, and two full keyword
definitions (Accelerate/疾行, Deflect/偏斜) — translated from Riot's official Core
Rules PDF (`playriftbound.com/en-us/rules-hub`), enough to exercise the full
pipeline end-to-end.

The full rules book is a larger, deliberate translation task (per the design doc,
this is meant to be an owned, terminology-consistent translation, not a fan or
machine one). To extend it, keep transcribing rule-by-rule in the same format
documented in the comment at the top of the file:

```
[<official rule id>] <optional short heading>
<body text — omit if the header line already carries the full rule text inline>
```

Rule IDs are copied straight from the official PDF's own numbering (e.g.
`103.2.d.1`), so the translation work maps 1:1 onto the source document.

After editing the Markdown, sync it into Postgres (upserts by `rule_id`; prunes
any row whose id no longer appears in the freshly parsed file):

```bash
uv run riftbound-rules-sync
```

## Card data

Card data lives in the `cards` Postgres table (~1,256 Traditional Chinese cards
from chroniclecore.com / 符文戰場編年史), populated by:

```bash
uv run python -m riftbound_bot.ingest.cards_scrape
```

This works by reading a single JSON payload the site's `/gallery` page embeds
client-side (Next.js RSC data) rather than requesting all 1,256 card pages — much
lighter on the site, but it also means it depends on that internal data shape and
**will break if the site changes it**. If it does, fall back to the English
community data source instead:

```bash
uv run python -m riftbound_bot.ingest.cards_from_api
```

This pulls the [Riftcodex](https://riftcodex.com) community REST API (English, no
auth, all 8 sets, ~1,131 cards after collapsing alternate-art/foil reprints) and
translates it via the configured generation model — one API call per 20 cards,
using whatever `GENERATION_*` settings are active.

Both scripts upsert by card `id`, so a re-run updates existing cards in place
rather than replacing the whole dataset.

## Build the vector index

```bash
uv run riftbound-build-index
```

Re-run this any time the `rules`/`cards` Postgres tables change. It's a full
rebuild (fixed snapshot, manual refresh, per the design doc — no watch/pipeline).
Nothing touches `data/turbovec/` until the rebuild finishes successfully: a crash
mid-build leaves the previous index completely untouched, never a partial one.

`build_index.py` also prepends ancestor-heading context to each chunk before
embedding it — see [retrieval-notes.md](retrieval-notes.md) for why.

## Open follow-ups

- Card data's top-level `tags` field (champion/region tags) comes from the source
  site un-localized — some entries are Simplified rather than Traditional Chinese.
- The rules corpus needs the rest of the Core Rules PDF transcribed (see above).
- Rule 809.1.c.1's worked example says "一張渾沌（Fury）法術" — 渾沌 means Chaos, not
  Fury (Fury is 狂怒, per the rune cards' data). Pre-existing typo.
- `data/cards/cards.json` is kept as a historical snapshot of the original scrape
  but is no longer read by any code path — safe to delete once you're comfortable
  relying on Postgres as the source of truth instead.
