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
  phrasing. When a force-included match's text uses a symbol (e.g. 811's
  "支付 [A]"), that symbol's defining rule is pulled in too, regardless of score.

Together these close the loop end-to-end for multi-hop questions like
"團結之印可以支付待命的費用嗎": 團結之印 (exact card match), 811 (exact keyword
match), and 135.2.e.5 — which contains 135.2.e.5.a, "[A] 可以任意屬性的力量支付" —
all land in the top-6 context together. Verified at the retrieval layer via
`RiftboundRagChain._retrieve()`.

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
