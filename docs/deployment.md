# Deployment

## Docker Compose

```bash
cp .env.example .env      # fill in the required values (see the README)
docker compose up -d
```

That's the whole deployment. The `bootstrap` service
([`ingest/bootstrap.py`](../src/riftbound_bot/ingest/bootstrap.py)) runs as an
init container ahead of `bot`: on a fresh checkout it syncs the rules Markdown
into Postgres, scrapes card data, and builds the vector index; on an already-built
deployment it checks for the index and exits immediately. `bot` waits on it via
`service_completed_successfully`, so it never starts against a missing index.

If bootstrap fails, `bot` won't start and the failure is in `docker compose logs
bootstrap` — that's deliberate. A bot serving an empty index looks healthy while
answering every question wrong.

The `bot` service still never connects to Postgres. It only reads the
`data/turbovec/` index, bind-mounted via `./data`; `postgres` is pulled in as
`bootstrap`'s dependency, not the bot's.

## Refreshing data

Bootstrap only fills in what's *missing* — it won't re-scrape or re-embed data
that's already there, since it re-runs on every `up`. Refreshing is deliberate,
manual work:

```bash
docker compose --profile tools run --rm ingest riftbound_bot.ingest.cards_scrape
docker compose --profile tools run --rm ingest riftbound_bot.ingest.rules_sync
docker compose --profile tools run --rm ingest riftbound_bot.ingest.build_index
docker compose restart bot
```

Pass the bare module path, not `python -m <module>`: the image's ENTRYPOINT is
already `python -m`, so the longer form runs `python -m python -m <module>` and
fails with `No module named python`.

## CI/CD

CI builds and pushes the image to GHCR on every push to `main`, then triggers a
redeploy via a webhook — see
[`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml).

Configure `ARCANE_WEBHOOK_URL` (or whatever your deploy webhook is) as a GitHub
Actions secret. That workflow file's comments cover the operational details: pull
policy, package visibility, and rollback.
