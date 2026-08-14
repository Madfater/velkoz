# Riftbound 規則 & 卡牌交互 Discord Bot

Traditional Chinese RAG-based Discord bot for Riftbound TCG rules Q&A and
card-interaction resolution.

Python + `discord.py` + LangChain, with generation and embeddings both going
through `langchain-openai` against OpenAI-compatible endpoints, TurboVec for the
local vector index, and PostgreSQL as the ingest-time source of truth. See
[docs/decisions.md](docs/decisions.md) for why each of those.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
cp .env.example .env           # fill in the values below
```

| Variable | Where to get it |
|---|---|
| `DISCORD_BOT_TOKEN` | [Discord Developer Portal](https://discord.com/developers/applications) → your app → Bot |
| `DISCORD_GUILD_ID` | Your server icon → Copy Server ID (enable Developer Mode first) |
| `EMBEDDING_API_BASE_URL` / `EMBEDDING_API_KEY` | your own self-hosted embeddings server — must serve an OpenAI-compatible `/embeddings` endpoint |
| `DATABASE_URL` | a running Postgres instance — `docker compose up -d postgres` locally, or your own |

Everything else in `.env.example` has a working default, including generation
(the free `opencode.ai/zen` gateway, no key needed).

Invite the bot with the `applications.commands` and `bot` scopes plus `Send
Messages`, `Create Public Threads`, `Send Messages in Threads`, and `Read Message
History`, and enable the **Message Content Intent** in the Developer Portal — it's
required to read follow-up messages inside threads.

## First run

```bash
docker compose up -d postgres                             # or point DATABASE_URL at your own
uv run python -m riftbound_bot.ingest.cards_scrape        # cards  → Postgres
uv run riftbound-rules-sync                               # rules  → Postgres
uv run riftbound-build-index                              # both   → data/turbovec/
uv run riftbound-bot
```

Then use `/ask` in your Discord server. The bot replies and spawns a thread from
its own reply — any further message in that thread is treated as a follow-up with
the prior Q&A as context.

## Tests

```bash
uv run pytest
```

No API keys needed — the RAG chain is tested against a stubbed retriever and LLM.

## Docs

- [docs/design.md](docs/design.md) — the original design doc: scope, data sources, non-goals.
- [docs/decisions.md](docs/decisions.md) — stack rationale and where the build deviates from that design.
- [docs/data-pipeline.md](docs/data-pipeline.md) — rules corpus format, card ingest, rebuilding the index.
- [docs/retrieval-notes.md](docs/retrieval-notes.md) — how retrieval works, plus a threshold that needs recalibrating.
- [docs/deployment.md](docs/deployment.md) — Docker Compose and the GHCR/webhook deploy.
