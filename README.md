# Riftbound 規則 & 卡牌交互 Discord Bot

Traditional Chinese RAG-based Discord bot for Riftbound TCG rules Q&A and card-interaction
resolution. Python + `discord.py` + LangChain, with generation and embeddings going through
OpenAI-compatible endpoints, [TurboVec](https://github.com/RyanCodrai/turbovec) for the local
vector index, and PostgreSQL as the ingest-time source of truth for card/rule data.

Further reading: [design & decisions](docs/riftbound-bot-design.md) ·
[retrieval behaviour & known issues](docs/retrieval-notes.md)

## Setup

```bash
uv sync --extra dev            # creates .venv from uv.lock
cp .env.example .env           # fill in the values below
```

Required in `.env`:

| Variable | Where to get it |
|---|---|
| `DISCORD_BOT_TOKEN` | [Discord Developer Portal](https://discord.com/developers/applications) → your app → Bot |
| `DISCORD_GUILD_ID` | Your server icon → Copy Server ID (enable Developer Mode in Discord settings first) |
| `EMBEDDING_API_BASE_URL` / `EMBEDDING_API_KEY` | your own self-hosted embeddings server — must serve an OpenAI-compatible `/v1/embeddings` endpoint |
| `DATABASE_URL` | a running Postgres instance — `docker compose up -d postgres` locally, or your own; ingest-time only, the live bot never connects to it |

`GENERATION_*` defaults to a free gateway and needs no key. Every other setting, and what it
does, is documented in [`.env.example`](.env.example).

Bot permissions needed when inviting it: `applications.commands`, `bot` scope with
`Send Messages`, `Create Public Threads`, `Send Messages in Threads`, `Read Message History`.
The **Message Content Intent** must be enabled for the bot in the Developer Portal (needed to
read follow-up messages inside threads).

## Ingest the data

```bash
uv run riftbound-cards-scrape    # cards from chroniclecore.com (zh-TW) into Postgres
uv run riftbound-rules-sync      # parses data/rules/*.md into Postgres
uv run riftbound-build-index     # reads both tables, builds data/turbovec/
```

`riftbound-cards-scrape` depends on an internal data shape of the source site and will break if
that site changes it; `uv run riftbound-cards-from-api` is the fallback, pulling the
[Riftcodex](https://riftcodex.com) community API in English and translating it via the
configured generation model. Both upsert by card `id`, so re-runs update in place.

Re-run `riftbound-build-index` after either — it's a full rebuild, and the live bot only ever
reads the built index, never Postgres.

## Running the bot

```bash
uv run riftbound-bot
```

Use `/ask` in your Discord server. The bot replies and spawns a thread from its own reply —
any further message in that thread is treated as a follow-up with the prior Q&A as context.

## Running with Docker

```bash
cp .env.example .env      # fill in the values above
docker compose build      # local image; the deploy host pulls from GHCR instead
docker compose up -d postgres
docker compose --profile tools run --rm ingest python -m riftbound_bot.ingest.cards_scrape
docker compose --profile tools run --rm ingest python -m riftbound_bot.ingest.rules_sync
docker compose --profile tools run --rm ingest python -m riftbound_bot.ingest.build_index
docker compose up -d bot
```

Postgres is only ever touched by the `ingest` profile — `bot` just needs restarting to pick up a
fresh index. The container runs as uid 1000 so it can write to the bind-mounted `./data`; if
your host user isn't uid 1000, adjust the `useradd`/`groupadd` lines in the Dockerfile or
`build_index` won't be able to write the index.

CI builds, pushes to GHCR and triggers a redeploy on every push to `main` — see
[`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml) for the operational details.

## Tests

```bash
uv run pytest
```

No API keys needed — the RAG chain is tested against a stubbed retriever and LLM.

The Postgres integration tests in `tests/test_db.py` are skipped unless `TEST_DATABASE_URL` is
set, because they **TRUNCATE** the `cards` and `rules` tables of whatever they connect to. Point
it at a scratch database — never at the `DATABASE_URL` holding your real ingested data:

```bash
TEST_DATABASE_URL=postgresql://riftbound:riftbound@localhost:5432/riftbound_test uv run pytest
```

If `TEST_DATABASE_URL` is set but unreachable, those tests **fail** rather than skip, so a
broken database service can't leave CI green with no SQL coverage.
