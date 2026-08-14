# Retrieval: as-built notes & known issues

What the retrieval layer actually does and what has been measured through it. The original
design intent lives in [`riftbound-bot-design.md`](riftbound-bot-design.md); this file records
where reality needed working around.

## Two pools, not one blended search

`RiftboundRagChain` searches rules and cards as two **separate** pools (each filtered by
`source_type`, `RETRIEVAL_POOL_PER_TYPE` candidates apiece) and merges by score, rather than one
blended top-k search. The card corpus outnumbers rules ~25 to 1 and is structurally homogeneous
(short, similarly-shaped strings) — a single combined search let cards drown out genuinely more
relevant rule chunks even when the rules-only pool ranked the correct answer #1 with a clean
score margin. See the docstring on `RiftboundRagChain` in
[`rag/chain.py`](../src/riftbound_bot/rag/chain.py).

`RETRIEVAL_SCORE_THRESHOLD` (default 0.5) gates each pool on embedding similarity — it's the
"nothing relevant was found" cutoff that produces the no-data reply instead of an answer
assembled from unrelated context.

TurboVec's relevance score is `(raw_cosine + 1) / 2`, so **0.5 means a raw cosine of 0**: it
discards results that are unrelated or actively anti-correlated, and nothing more. The previous
default of 0.45 was a Chroma-era raw-cosine value; on TurboVec's scale it corresponds to a
cosine of ≈ −0.1, which every result clears — so the safety cutoff never fired at all.

0.5 is a floor, not a calibration. To tune it for your own embedding endpoint, run with
`LOG_LEVEL=DEBUG` and compare the `score_min`/`score_max` the chain logs for a known-relevant
question against a clearly off-topic one, then set the threshold between the two. That is a
measurement through the real `vectorstore.py`/`chain.py` path, which is what the bot calls —
not an isolated script.

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

One lead is closed: the distance metric is not the cause. TurboVec computes cosine similarity
unconditionally in the pinned release, with no `similarity=` kwarg to set (see the comment in
`vectorstore.py`). The underlying embedding-quality gap for non-exact card queries is still
open. Since then, card documents also embed colour/rarity/stats/tags, which gives
attribute-shaped questions something to match on — but that is added retrievable surface, not a
fix for the similarity ranking itself.

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

## Open follow-ups

- Card data's top-level `tags` field (champion/region tags) comes from the source site
  un-localized — some entries are Simplified rather than Traditional Chinese.
- The rules corpus holds only a sample slice of the Core Rules PDF; the rest still needs
  transcribing, in the format documented at the top of `data/rules/core_rules_zh_tw.md`.
- Rule 809.1.c.1's worked example says "一張渾沌（Fury）法術" — 渾沌 means Chaos, not Fury (Fury is
  狂怒, per the rune cards' data). Pre-existing typo, not touched by this pass.
