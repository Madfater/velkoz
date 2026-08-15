# Data pipeline

**There is no local data directory.** Everything lives in Postgres — three
tables, all in [`ingest/db.py`](../src/riftbound_bot/ingest/db.py):

```
rules        # the hand-translated rules corpus (narrow + JSONB)
cards        # scraped card data (narrow + JSONB)
embeddings   # the vector index the bot serves from (pgvector)
```

`cards_scrape.py`/`cards_from_api.py` upsert scraped cards into `cards`
(per-record, not a whole-file replace), `rules_import.py` parses Markdown into
`rules`, and `build_index.py` reads both tables and writes `embeddings`.

Unlike earlier versions, the live bot reads Postgres directly — the `embeddings`
table, on every request. See `PgVectorStore` in
[`rag/vectorstore.py`](../src/riftbound_bot/rag/vectorstore.py).

Two corpora ship inside the package, at
[`ingest/seeds/`](../src/riftbound_bot/ingest/seeds/), so a fresh environment can
initialize with no manual step and no reachable third-party site. They are
*seeds*, not the source of truth: bootstrap reads one only into an empty table
and never over existing rows.

The sections below describe each step individually. You don't have to run them
by hand on a fresh deployment:
[`ingest/bootstrap.py`](../src/riftbound_bot/ingest/bootstrap.py) chains them in
order (rules → cards → build index) and is what `docker compose up` runs before
starting the bot. It only fills in what's missing — it skips rules and cards when
those tables are populated, and does nothing at all once `embeddings` has rows —
so refreshing data that's already there is still a deliberate manual run of the
steps below.

## Rules corpus

**The `rules` table is the source of truth.** The seed at
`src/riftbound_bot/ingest/seeds/core_rules_zh_tw.md` currently holds a **sample
slice** — the Golden/Silver Rules, the start of deck construction, and two full
keyword definitions (Accelerate/疾行, Deflect/偏斜) — translated from Riot's
official Core Rules PDF (`playriftbound.com/en-us/rules-hub`), enough to exercise
the full pipeline end-to-end. It initializes an empty database and is never
re-applied over rules already stored.

The full rules book is a larger, deliberate translation task (per the design doc,
this is meant to be an owned, terminology-consistent translation, not a fan or
machine one). To extend it, keep transcribing rule-by-rule in the same format
documented in the comment at the top of the seed file:

```
[<official rule id>] <optional short heading>
<body text — omit if the header line already carries the full rule text inline>
```

Rule IDs are copied straight from the official PDF's own numbering (e.g.
`103.2.d.1`), so the translation work maps 1:1 onto the source document.

Markdown stays the editing format, but it's an interchange now rather than a
directory the deployment reads — so the CLIs take explicit paths. Export what's
stored, edit it, import it back (upserts by `rule_id`; prunes any row whose id no
longer appears in the imported file):

```bash
uv run riftbound-rules-export -o rules.md
$EDITOR rules.md
uv run riftbound-rules-import rules.md
```

Pass `--no-prune` when importing a *partial* corpus — otherwise the import is
treated as the complete set and everything absent from it is deleted.

Because the database is the live copy, **back it up**. `riftbound-rules-export`
is the reviewable, diffable form of that; `pg_dump` covers everything. To update
the shipped seed after editing, export over it and re-add the `<!-- -->` header
comment, which documents the authoring convention and doesn't survive the round
trip (the parser strips comments, so they never reach Postgres).

## Card data

Card data lives in the `cards` Postgres table (~1,256 Traditional Chinese cards
from chroniclecore.com / 符文戰場編年史), populated by:

```bash
uv run python -m riftbound_bot.ingest.cards_scrape
```

This works by reading a single JSON payload the site's `/gallery` page embeds
client-side (Next.js RSC data) rather than requesting all 1,256 card pages — much
lighter on the site, but it also means it depends on that internal data shape and
**will break if the site changes it**.

When it does, bootstrap falls back to the snapshot at
`src/riftbound_bot/ingest/seeds/cards.json` so a fresh environment still comes
up — stale, but working and free. For genuinely fresher data, the English
community source is the manual fallback:

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
The rebuild writes to a side table and renames it over `embeddings` at the very
end, so a crash mid-build leaves the previous index completely untouched and
still being served, never a partial one.

The `vector(N)` column is sized from the first batch of vectors the embedding
endpoint actually returns — nothing configures the dimension, so a model swap
just needs a rebuild rather than a setting kept in lockstep.

`build_index.py` also prepends ancestor-heading context to each chunk before
embedding it — see [retrieval-notes.md](retrieval-notes.md) for why.

## Card images

Card records carry an `image_url` — the card's face, which `/card` renders. It is
read from the gallery payload's `assets.img_zh_hans`, **not** templated from the
card id: 45 of the 1,256 cards have a filename the obvious
`unified/{id}_sc.png` template gets wrong (ids ending in `*` are spelled `_star_`,
so `VEN-189*` is `unified/VEN-189_star_sc.png`).

It uses the `zh_hans` face because that is the only one present for all 1,256
cards — `img_en` is missing for 68, mostly VEN promos and rune cards. The printed
text on it is therefore Simplified while the rest of the bot is Traditional. That
trade was made deliberately: an image that always resolves beats a matching script
that sometimes leaves a hole where the card should be.

`image_url` is deliberately absent from `CardRecord.text`, the string that gets
embedded, so adding it did not invalidate a single vector.

Cards stored before this field existed have no image, and the bot refuses to start
on them rather than serving card embeds with a blank space. `bootstrap` repairs
that automatically on the next boot (`_ensure_card_images`), so a normal `compose
up` is enough; to do it by hand:

```bash
docker compose --profile tools run --rm ingest riftbound_bot.ingest.cards_scrape
```

## Open follow-ups

- Re-scraping cards updates `cards` but not `embeddings`, so card-name corrections
  reach `/card` immediately and `/ask` only after a `build_index` rebuild. The
  2026-08 re-scrape renamed 19 cards (提摩 → 提摩-偵察兵) and recategorized 18
  (指示物單位 → 衍生物單位); those are live for `/card` and stale for retrieval
  until the index is rebuilt.
- The rules corpus needs the rest of the Core Rules PDF transcribed (see above).
- Rule 809.1.c.1's worked example says "一張渾沌（Fury）法術" — 渾沌 means Chaos, not
  Fury (Fury is 狂怒, per the rune cards' data). Pre-existing typo.
- The card snapshot at `src/riftbound_bot/ingest/seeds/cards.json` drifts further
  from the live site with every set. Re-running `cards_scrape` and re-exporting it
  keeps the offline fallback useful. Last refreshed 2026-08-15, which is also when
  the source site's `tags` switched to Traditional Chinese — the un-localized tags
  noted here previously are fixed as of that snapshot.
