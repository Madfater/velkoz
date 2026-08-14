# Riftbound Rules & Card-Interaction Discord Bot — Design

## Context

A Discord bot that clarifies Riftbound (Riot Games' TCG) rules and explains what happens when a specific card is played, for a Traditional Chinese–speaking community. Two motivations: a personal project to learn RAG (retrieval-augmented generation) hands-on, and a real gap — no existing Riftbound bot supports Chinese at all.

## Scope

- **Unified assistant**: general rules clarification ("what does keyword X mean") and specific card-interaction resolution ("what happens if I play X while opponent has Y") are handled by the same bot, not split into separate tools.
- **Single Discord server** the builder controls — not a public multi-server bot. Simpler permissions/rate-limiting; opening it up publicly is a later decision, not part of this spec.
- **Traditional Chinese only.** Simplified Chinese was considered and dropped — narrowing scope on purpose.
- **Stateless / abstract.** The bot does not track an ongoing game's board state. Users describe the relevant state in their question ("I have X in play, opponent has Y, I cast Z"). A full game-state tracker was explicitly ruled out as a much bigger project.
- **Fixed data snapshot, manual refresh.** No automated ingestion pipeline for new sets/errata — the game updates slowly enough that this isn't worth building for v1.

## Architecture

- **RAG pipeline built with LangChain** (the builder's choice — a hand-built pipeline was suggested as more educational for learning RAG mechanics directly, but LangChain was preferred).
- **LLM: Claude API.** DeepSeek was seriously considered for its much lower cost (~7–20x cheaper than Claude at comparable tiers) and strong Chinese-language fluency, but rejected because DeepSeek's own documentation and third-party evaluations report notable hallucination rates and occasionally malformed tool-call JSON — directly at odds with the strict-grounding requirement below. DeepSeek also processes/stores data on mainland China servers and its ToS reportedly permits training on API data, which may matter given the bot's Taiwan/Hong Kong-facing audience. **Revisit DeepSeek later if cost becomes a real constraint** — LangChain makes the provider swap a config change, not a rewrite.
- **Strict grounding**: answers must be grounded in retrieved text with citations back to specific rules. No free-form reasoning beyond what's retrieved — a confident wrong answer is worse than a slower, cited one for a rules-adjudication tool.
- **Interaction model**: `/ask` slash command starts a conversation. The bot replies inside a **Discord thread** spawned from its own message; any message posted in that thread is treated as a follow-up with the prior Q&A as context. This avoids polluting the main channel and gives an unambiguous signal for "this is part of the conversation" without hand-rolled session tracking.

## Data sources

**Rules text**: Riot's official English Core Rules PDF (`playriftbound.com/en-us/rules-hub`) — the only rules-authoritative source; no official Chinese translation exists (confirmed — only unofficial fan translations circulate: a DeepSeek machine-translated Simplified version on 旅法師營地/iyingdi.com, and an unofficial Traditional Chinese fan wiki). The builder will **self-translate the English Core Rules once**, producing an owned, terminology-consistent Traditional Chinese corpus, rather than adopting either fan translation wholesale. Those fan translations are useful only as a spot-check reference for terminology.

**Card data**: scraped from **chroniclecore.com** (符文戰場編年史, a Taiwan fan site with ~1,256 Traditional Chinese cards, English cross-reference, keyword/filter search). No documented API or bulk export exists — this is a browsable web UI, scraped at the builder's own risk (unofficial site, reuse terms unclear, personal/non-commercial use). **Fallback**: if scraping proves infeasible or the site changes, self-translate English card data the same way the rules text is being handled. This originally used `OwenMelbz`'s GitHub Gist, but that turned out to be a one-revision snapshot frozen at 2025-11-15 (394 cards, three sets behind), so the fallback now reads the **Riftcodex** community REST API (`api.riftcodex.com`, no auth, all 8 sets, tracked Vendetta within days of its 2026-07-31 release). Riot's own gallery at playriftbound.com was considered and rejected: it ships no Chinese locale, and scraping its Next.js payload would make the backup fail the same way as the primary.

**Why not Riot's official developer API**: it explicitly prohibits applications that "automate rules/interactions/resolutions" — exactly this project's function. The public Core Rules PDF plus community/fan data is the practical, non-restricted path for a personal project.

## Positioning

RiftJudge (riftjudge.com) already offers English Riftbound rules Q&A as a Discord bot (`!ask` command, freemium, `!learn` correction loop). Its source doesn't appear to be public (no repo found, though not explicitly stated closed). No existing bot — RiftJudge or the smaller open-source card-lookup bots — supports Chinese in any form. This project is not a RiftJudge clone; it fills a real, currently-unserved niche.

## Explicitly deferred (not gaps, deliberate non-goals for v1)

- Live game/board-state tracking
- Automated data-ingestion pipeline for new sets/errata
- Public multi-server distribution
- Simplified Chinese support
