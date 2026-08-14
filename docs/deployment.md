# Deployment

## Docker Compose

```bash
cp .env.example .env      # fill in the required values (see the README)
docker compose up -d
```

That's the whole deployment. There is no data directory to create, own, or
keep in sync — everything lives in Postgres, and the rules and card corpora
ship inside the image
([`ingest/seeds/`](../src/riftbound_bot/ingest/seeds/)).

The `bootstrap` service
([`ingest/bootstrap.py`](../src/riftbound_bot/ingest/bootstrap.py)) runs as an
init container ahead of `bot`: on a fresh database it seeds the rules, loads card
data, and builds the vector index; on an already-built deployment it checks for
the index and exits immediately. `bot` waits on it via
`service_completed_successfully`, so it never starts against a missing index.

A fresh environment needs no manual step and no reachable third-party site: if
the card scrape breaks, bootstrap falls back to the bundled snapshot.

If bootstrap fails, `bot` won't start and the failure is in `docker compose logs
bootstrap` — that's deliberate. A bot serving an empty index looks healthy while
answering every question wrong.

Unlike earlier versions, `bot` connects to Postgres itself: it reads the
`embeddings` table on every request, so `postgres` is its own dependency and not
merely `bootstrap`'s.

## Backups

Postgres now holds the only live copy of the hand-translated rules. The shipped
seed is a snapshot, not a backup of anything edited since — so back the database
up, or export the translation to reviewable Markdown:

```bash
docker compose --profile tools run --rm ingest riftbound_bot.ingest.rules_export > rules.md
```

## Refreshing data

Bootstrap only fills in what's *missing* — it won't re-scrape or re-embed data
that's already there, since it re-runs on every `up`. Refreshing is deliberate,
manual work:

```bash
docker compose --profile tools run --rm ingest riftbound_bot.ingest.cards_scrape
docker compose --profile tools run --rm ingest riftbound_bot.ingest.build_index
docker compose restart bot
```

To change the rules themselves, edit Markdown and import it (the path is an
argument now — there is no configured rules directory):

```bash
docker compose --profile tools run --rm ingest riftbound_bot.ingest.rules_import /path/to/rules.md
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
