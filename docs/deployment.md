# Deployment

## Docker Compose

```bash
cp .env.example .env      # fill in the required values (see the README)
docker compose up -d postgres
docker compose --profile tools run --rm ingest python -m riftbound_bot.ingest.cards_scrape
docker compose --profile tools run --rm ingest python -m riftbound_bot.ingest.rules_sync
docker compose --profile tools run --rm ingest python -m riftbound_bot.ingest.build_index
docker compose up -d bot
```

The `postgres` service is only ever touched by the `ingest` profile (`build_index`
and the scrape/sync scripts). The `bot` service never connects to it — only to the
`data/turbovec/` index that `build_index` produces, bind-mounted via `./data`.

Re-run the `ingest` steps any time the rules Markdown or card data changes; `bot`
just needs restarting to pick up a fresh index.

## CI/CD

CI builds and pushes the image to GHCR on every push to `main`, then triggers a
redeploy via a webhook — see
[`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml).

Configure `ARCANE_WEBHOOK_URL` (or whatever your deploy webhook is) as a GitHub
Actions secret. That workflow file's comments cover the operational details: pull
policy, package visibility, and rollback.
