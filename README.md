# Riftbound 規則 & 卡牌交互 Discord Bot

Traditional Chinese RAG-based Discord bot for Riftbound TCG rules Q&A and card-interaction
resolution. See [`riftbound-bot-design.md`](riftbound-bot-design.md) for the full design doc.

## Stack

Python + `discord.py` + LangChain, with both generation and embeddings going through
`langchain-openai` pointed at OpenAI-compatible endpoints (rather than a single fixed provider)
plus `langchain-chroma` for local vector storage.

**This deviates from the design doc's original choice of Claude for generation.** The doc
picked Claude specifically over DeepSeek for grounding/hallucination reasons — DeepSeek's
higher hallucination rate is exactly what a strict-grounding rules-adjudication tool can least
afford. The current default (`deepseek-v4-flash-free` via the free `opencode.ai/zen` gateway)
trades that quality margin for $0 cost. `GENERATION_API_BASE_URL`/`GENERATION_MODEL` are a
config change, not a rewrite, so switching back to Claude (or anything else OpenAI-API-shaped)
is one env var edit — see `.env.example`.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
cp .env.example .env           # fill in the values below
```

Required in `.env`:

| Variable | Where to get it |
|---|---|
| `DISCORD_BOT_TOKEN` | [Discord Developer Portal](https://discord.com/developers/applications) → your app → Bot |
| `DISCORD_GUILD_ID` | Your server icon → Copy Server ID (enable Developer Mode in Discord settings first) |
| `EMBEDDING_API_BASE_URL` / `EMBEDDING_API_KEY` | your own self-hosted embeddings server — must serve an OpenAI-compatible `/embeddings` endpoint |

`GENERATION_API_BASE_URL`/`GENERATION_MODEL` default to the free `opencode.ai/zen` gateway
(`deepseek-v4-flash-free`) — no key needed, `GENERATION_API_KEY` can stay blank. The same
gateway also serves Claude and other frontier models under that base URL (paid, unlike the
DeepSeek default) — set `GENERATION_MODEL=claude-sonnet-5` to switch back to what the design
doc originally specified without touching any code. Any other OpenAI-API-compatible provider
works the same way by pointing `GENERATION_API_BASE_URL` at it instead.

Bot permissions needed when inviting it: `applications.commands`, `bot` scope with
`Send Messages`, `Create Public Threads`, `Send Messages in Threads`, `Read Message History`.
The **Message Content Intent** must be enabled for the bot in the Developer Portal (needed to
read follow-up messages inside threads).

## Data pipeline

```
data/rules/core_rules_zh_tw.md   # self-translated Traditional Chinese rules corpus
data/cards/cards.json            # normalized card data
data/chroma/                     # built vector index (gitignored, rebuild with build_index)
```

### Rules corpus

`data/rules/core_rules_zh_tw.md` currently holds a **sample slice** — the Golden/Silver Rules,
the start of deck construction, and two full keyword definitions (Accelerate/疾行,
Deflect/偏斜) — translated from Riot's official Core Rules PDF
(`playriftbound.com/en-us/rules-hub`), enough to exercise the full pipeline end-to-end.

The full rules book is a larger, deliberate translation task (per the design doc — this is
meant to be an owned, terminology-consistent translation, not a fan/machine one). To extend it,
keep transcribing rule-by-rule in the same format documented in the comment at the top of the
file:

```
[<official rule id>] <optional short heading>
<body text — omit if the header line already carries the full rule text inline>
```

Rule IDs are copied straight from the official PDF's own numbering (e.g. `103.2.d.1`), so the
translation work maps 1:1 onto the source document.

`build_index.py` prepends each chunk's top-level section text (e.g. "805 疾行（Accelerate）")
to its embedded content — short sub-rules like "805.1" are often just a few words
("疾行是一種單位能力。") and embed almost meaninglessly without their keyword name attached,
confirmed by checking real retrieval rankings before landing on this.

### Card data

`data/cards/cards.json` is **already the complete dataset** — all 1,256 Traditional Chinese
cards from chroniclecore.com (符文戰場編年史), scraped via:

```bash
uv run python -m riftbound_bot.ingest.cards_scrape data/cards/cards.json
```

This works by reading a single JSON payload the site's `/gallery` page embeds client-side
(Next.js RSC data) rather than requesting all 1,256 card pages — much lighter on the site, but
it also means it depends on that internal data shape and **will break if the site changes it**.
If it does, fall back to the English community data source instead:

```bash
uv run python -m riftbound_bot.ingest.cards_from_gist data/cards/cards.json
```

This pulls OwenMelbz's community card-data gist and translates it via the configured generation
model — it costs one API call per 20 cards, using whatever `GENERATION_*` settings are active.

### Build the vector index

Re-run this any time `data/rules/` or `data/cards/cards.json` changes — it's a full rebuild
(fixed snapshot, manual refresh, per the design doc; no watch/pipeline):

```bash
uv run riftbound-build-index
```

## Running the bot

```bash
uv run riftbound-bot
```

Use `/ask` in your Discord server. The bot replies and spawns a thread from its own reply —
any further message in that thread is treated as a follow-up with the prior Q&A as context.

## Tests

```bash
uv run pytest
```

Covers rule-corpus parsing, citation formatting, and the RAG chain's prompt/citation wiring
against a stubbed retriever + LLM — no API keys needed. `build_index` and real retrieval need
your embeddings endpoint reachable; getting real *answers* end-to-end also needs your
generation endpoint reachable and, for the live bot, a Discord bot token.

### Retrieval design note

`RiftboundRagChain` searches rules and cards as two **separate** pools (each filtered by
`source_type`, `RETRIEVAL_POOL_PER_TYPE` candidates apiece) and merges by score, rather than one
blended top-k search. The card corpus outnumbers rules ~25 to 1 and is structurally homogeneous
(short, similarly-shaped strings) — a single combined search let cards drown out genuinely more
relevant rule chunks even when the rules-only pool ranked the correct answer #1 with a clean
score margin. See the docstring on `RiftboundRagChain` in `rag/chain.py`.

`RETRIEVAL_SCORE_THRESHOLD` (default 0.45) gates each pool on embedding cosine similarity —
it's the "nothing relevant was found" safety cutoff before the LLM is even called. Sanity-check
this against your own embedding endpoint before trusting the default: retrieve a known-relevant
query and a clearly off-topic one, and confirm there's an actual, reproducible score gap between
them (not just in an isolated script — through `vectorstore.py`'s real `similarity_search_with_
relevance_scores`, which is what the bot actually calls).

## Known issue: card retrieval quality is unresolved for non-exact queries

Even with the separate-pool fix above, asking about a specific card **by its exact name**
("阿璃-誘人這張卡是什麼效果？" / what does Ahri, Alluring do?) failed to surface that card in
the top 6 — it ranked 661st out of 1,256 cards, behind unrelated cards, through the real
`vectorstore.py` code path (not a one-off script).

**Exact-full-name queries are now fixed**, but via a targeted workaround rather than an
embedding-quality fix: `RiftboundRagChain` (`rag/chain.py`) loads every indexed card's `name_zh`
once at construction and, per query, force-includes any card whose full name appears literally
in the question — bypassing similarity scoring entirely for that case. This does **not** help
fuzzy/partial/misspelled names or pure keyword-effect queries with no literal card name in the
text (e.g. "哪些卡可以獲得黃色力量？") — those still rely entirely on embedding similarity and
remain unaddressed. Likely causes for the underlying embedding-quality gap, still untested: the
embedding model handling short/templated card text differently than prose rules text; how the
self-hosted endpoint batches large embedding requests. Rules-only retrieval looked much healthier
in isolated testing but wasn't confirmed through that same production path before this was set
aside.

~~One concrete, unverified lead: `vectorstore.py` never sets a distance metric~~ — resolved:
`vectorstore.py` sets `collection_configuration={"hnsw": {"space": "cosine"}}`, confirmed live
against the real index that the collection's stored config actually uses cosine, not the
Euclidean default. That specific lead is closed; the underlying embedding-quality gap for
non-exact card queries is not.

**The same problem was independently confirmed for rule keyword sections** on multi-hop
interaction questions (e.g. "團結之印可以支付待命的費用嗎") — a keyword's decisive sub-rule can
rank outside the pool even with `build_index.py`'s topic-context prepending. `_exact_keyword_matches`
in `rag/chain.py` applies the identical exact-substring workaround to rule sections shaped like
card names (a top-level `[NNN] CJK term（English gloss）` title, e.g. 811 待命（Hidden）), detected
structurally rather than via a hardcoded keyword list.

A sub-rule buried several levels under a broad section (e.g. `135.2.e.5.a`, under `[135] 規則文字
（Rules Text）` → `[135.2.e] 符號`) can still fail to surface on similarity search alone even after
`build_index.py`'s topic-context prepending was generalized to walk the *whole* ancestor-heading
chain (not just the top-level one — see its docstring for the heading-vs-rule-statement heuristic,
since every rule in this corpus is written as `[id] text` on one line, so a naive "has a body"
check can't tell the two apart). That change is real, correctly scoped (a no-op for 805/809/811,
already shallow), and genuinely improves 135.2.e.5.a's embedded context — but confirmed live its
similarity score barely moved (0.366 → 0.373, still ~64th/94), nowhere near the top-6 pool. Same
underlying embedding-quality gap already documented above for cards, independently reconfirmed for
rules: no amount of context-prepending fixes a similarity model that doesn't rank a specific
sub-rule above its own broader ancestor section for a multi-hop question.

**Closed via a different mechanism**: `_symbol_expansions` in `rag/chain.py` does a one-hop
expansion over this corpus's small, closed set of bracket shorthand symbols ([A], [C], [M], [E],
[R]/[G]/[B]/[O]/[P]/[Y]) — detected structurally via the corpus's own "簡稱為 [X]" phrasing, not a
hardcoded table. When a force-included exact match's text uses a symbol (e.g. 811's "支付 [A]"),
that symbol's own defining rule is pulled in too, regardless of its similarity score. Confirmed
live this closes the loop for the 團結之印/待命 question end-to-end at the retrieval layer: 團結之印
(exact card match), 811 (exact keyword match), and 135.2.e.5 — which contains 135.2.e.5.a, "[A]
可以任意屬性的力量支付" — all land in the top-6 context together. (The free generation endpoint hit
its rate limit while verifying the final LLM-generated answer text; the retrieval-side result is
deterministic and independently confirmed via `RiftboundRagChain._retrieve()`, not re-run at
generation time.)

## Known follow-ups

- `data/cards/cards.json`'s top-level `tags` field (champion/region tags) comes from the source
  site un-localized — some entries are Simplified rather than Traditional Chinese.
- The rules corpus needs the rest of the Core Rules PDF transcribed (see above).
- Rule 809.1.c.1's worked example says "一張渾沌（Fury）法術" — 渾沌 means Chaos, not Fury (Fury is
  狂怒, per `data/cards/cards.json`'s rune cards). Pre-existing typo, not touched by this pass.
