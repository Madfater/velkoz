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

CI builds and pushes the image to GHCR on every push to `main`, then deploys it
to Arcane — see [`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml).

Configure two GitHub Actions secrets: `ARCANE_URL` (the instance base URL) and
`ARCANE_API_KEY` (Settings -> API Keys in Arcane; sent as `X-Api-Key`).

Deploying is **two** API calls, in order — sync the compose from git, then bring
the project up. Both matter. The Arcane project is GitOps-managed but configured
`autoSync: false`, so nothing pulls a new `docker-compose.yml` on its own: skip
the sync call and Arcane will happily redeploy the compose it last saw while the
image moves on underneath it. That mismatch is not a hypothetical — it is how the
deploy broke on 2026-08-14, and the symptom was a bot that never started at all
(`bootstrap` exited 1 on `CREATE EXTENSION vector` against a stale
`postgres:16-alpine`). The older webhook-based trigger only ever did the second
call, which is why it could not have caught this.

The workflow file's comments cover the rest: pull policy, package visibility, and
rollback.
