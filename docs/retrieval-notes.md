# Retrieval notes

How `RiftboundRagChain` finds context, what's been worked around, and what's
still broken. The authoritative detail lives in the docstrings in
[`rag/chain.py`](../src/riftbound_bot/rag/chain.py); this is the overview.

## Design

### Separate pools per source type

`RiftboundRagChain` searches rules and cards as two **separate** pools (each
filtered by `source_type`, `RETRIEVAL_POOL_PER_TYPE` candidates apiece) and
merges by score, rather than one blended top-k search.

The card corpus outnumbers rules ~25 to 1 and is structurally homogeneous (short,
similarly-shaped strings). A single combined search let cards drown out genuinely
more relevant rule chunks even when the rules-only pool ranked the correct answer
#1 with a clean score margin.

### Ancestor-heading context prepending

`build_index.py` prepends each chunk's ancestor-heading chain to its embedded
content. Short sub-rules like "805.1" are often just a few words
("疾行是一種單位能力。") and embed almost meaninglessly without their keyword name
attached — confirmed by checking real retrieval rankings before landing on this.

Every rule in this corpus is written as `[id] text` on one line, so a naive "has a
body" check can't tell a heading from a rule statement; see `build_index.py`'s
docstring for the heuristic that does.

### Exact-match escape hatches

Three mechanisms in `chain.py` bypass similarity scoring entirely for cases where
it demonstrably fails. All three detect their targets structurally — no hardcoded
keyword or symbol tables:

- **`_exact_name_matches`** — loads every indexed card's `name_zh` once at
  construction and force-includes any card whose full name appears literally in
  the question.
- **`_exact_keyword_matches`** — the same mechanism for rule sections shaped like
  card names (a top-level `[NNN] CJK term（English gloss）` title, e.g.
  `811 待命（Hidden）`).
- **`_symbol_expansions`** — one-hop expansion over the corpus's small, closed set
  of bracket shorthand symbols (`[A]`, `[C]`, `[M]`, `[E]`,
  `[R]`/`[G]`/`[B]`/`[O]`/`[P]`/`[Y]`), detected via the corpus's own "簡稱為 [X]"
  phrasing. When a retrieved chunk's text uses a symbol (e.g. 811's "支付 [A]"),
  that symbol's defining rule is pulled in too, regardless of score. This runs
  over every chunk heading for the prompt, the similarity-ranked ones included —
  the answer must gloss each symbol on first use (see below), and a symbol
  carried in by a pool hit needs its definition just as much as one carried in
  by an exact match.

Together these close the loop end-to-end for multi-hop questions like
"團結之印可以支付待命的費用嗎": 團結之印 (exact card match), 811 (exact keyword
match), and 135.2.e.5 — which contains 135.2.e.5.a, "[A] 可以任意屬性的力量支付" —
all land in the top-6 context together. Verified at the retrieval layer via
`RiftboundRagChain._retrieve()`.

## Citations are `[來源N]`, never bare `[N]`

Three places agree on this format and have to move together: `SYSTEM_PROMPT`
rule 2, the context-block prefix in `ask()`, and `format_citations`.

Bare `[N]` was ambiguous, not merely ugly. The rules corpus writes generic energy
costs as bracketed digits — 805.1.a has `額外支付 [1][C]`, 135.2.e.6.c has
`費用為 [1][C]` —
so a `[1]` in an answer could equally mean "source 1" or "1 generic energy", and
the same collision hit the model reading its own context block. Domain symbols
(`[A]`, `[C]`) had the same problem one letter over.

`format_citations` numbers the footer by each entry's **position in `metadatas`**,
which is what the context block numbered, not by the position of the rendered
line. It de-duplicates by label, and a label reached from two retrieval slots
keeps both markers (`[來源2][來源5] 規則 805.1`). Numbering rendered lines instead
would shift every entry below a collapse onto a number belonging to a different
source — silently wrong in a way the reader cannot detect.

## Answers gloss their own jargon

`SYSTEM_PROMPT` rule 7 requires the first mention of any symbol or game term to
carry a short parenthetical gloss taken *from the retrieved content*, and to be
marked "（檢索內容未收錄此術語的定義）" when the retrieval doesn't define it. The
second half matters more than it looks: it is what surfaces corpus gaps to the
reader instead of leaving bare jargon in a ruling.

There are real gaps behind it. Rule 811.1.b alone references 英雄區域, 開放狀態,
反應, 符文池 and 傳奇區 — every one of them a passing mention with no definition
anywhere in the corpus, because the seed is a documented sample slice and the
zones chapter was never ingested. Ingesting it is the actual fix; rule 7b is the
honest stopgap.

**`[1]` still cannot be glossed.** `_SYMBOL_RE` and `_SYMBOL_DEFINITION_RE` match
`[A-Za-z]` only, and no rule defines `[1]`. Closing that needs both a widening to
`[A-Za-z0-9]` and a new `簡稱為 [1]`-shaped corpus line, plus a reindex.

## Rules and cards must use the same words

The rules corpus is hand-translated; card text is Riot's own zh-TW, scraped. When
the two disagree, retrieval breaks silently — a player types what is printed on
their card and matches nothing.

This has bitten once already: the seed rendered *Champion* as 冠軍 (冠軍區,
冠軍傳奇) while all 1,256 cards use 英雄 (`英雄單位` as a card type, `英雄區域` in
card text) and never once say 冠軍. A question phrased with 英雄區域 could not
reach rule 811.1.b at all.

The seed is corrected, which is enough for a **fresh** environment. It is not
enough for one that has already bootstrapped: the `rules` table is the source of
truth and the seed is never re-applied over stored rows (see
[data-pipeline.md](data-pipeline.md)), so an existing deployment keeps serving
冠軍 until someone runs the round trip and rebuilds the index:

```bash
uv run riftbound-rules-export -o rules.md   # then apply 冠軍 → 英雄
uv run riftbound-rules-import rules.md
uv run python -m riftbound_bot.ingest.build_index
```

**A known live instance of the same class:** Might is `[M]` in the rules, bare `S`
in scraped card text (327 occurrences), and `戰力` in `cards_from_api.py`'s
translation prompt. Three vocabularies, one concept. When adding rules text, check
the term against `cards.json` first.

## Action required: recalibrate the score threshold

`RETRIEVAL_SCORE_THRESHOLD` (default 0.45) gates each pool on embedding cosine
similarity — it's the "nothing relevant was found" safety cutoff before the LLM is
even called.

**This default was calibrated against Chroma's raw cosine similarity and has
never been re-checked since.** Both stores since have deliberately kept the same
`(raw_cosine + 1) / 2` mapping — TurboVec by its own convention, pgvector via
`_relevance` in `vectorstore.py` — so the migrations didn't move the goalposts,
but they didn't fix this either: 0.45 on that scale is a raw cosine of only
≈ −0.1, far too permissive.

To recalibrate: retrieve a known-relevant query and a clearly off-topic one, and
confirm there's an actual, reproducible score gap between them. Do this through
`vectorstore.py`'s real `similarity_search_with_relevance_scores` — the path the
bot actually calls — not an isolated script.

## Open: embedding quality for non-exact queries

The exact-match workarounds above are targeted patches, not an embedding-quality
fix. The underlying gap is unresolved.

**What's still unaddressed:**

- Fuzzy, partial, or misspelled card names.
- Pure keyword-effect queries with no literal card name in the text (e.g.
  "哪些卡可以獲得黃色力量？").
- Deeply nested sub-rules. `135.2.e.5.a` (under `[135] 規則文字（Rules Text）` →
  `[135.2.e] 符號`) still scores ~64th/94 (0.366 → 0.373) even after
  ancestor-chain prepending was generalized to walk the whole chain rather than
  just the top-level heading. That change is real and correctly scoped, but no
  amount of context-prepending fixes a similarity model that won't rank a specific
  sub-rule above its own broader ancestor section for a multi-hop question.

For scale on how bad the raw ranking can be: before the exact-name workaround,
asking about a card **by its exact full name** ("阿璃-誘人這張卡是什麼效果？")
ranked it 661st out of 1,256 — behind unrelated cards, through the real
`vectorstore.py` code path, not a one-off script.

**Untested hypotheses:** the embedding model handling short/templated card text
differently than prose rules text; how the self-hosted endpoint batches large
embedding requests. Rules-only retrieval looked much healthier in isolated testing
but wasn't confirmed through that same production path before this was set aside.
