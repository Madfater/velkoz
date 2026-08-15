"""CLI: brings a deployment up to a servable state, then exits.

`docker compose up` runs this before the bot starts (see the `bootstrap`
service in docker-compose.yml), so a clean checkout produces a working bot
instead of one that crash-loops on a missing index until three ingest
commands are run by hand in the right order.

A fresh environment needs no manual step and no reachable third-party site:
rules and cards both fall back to the snapshots in ingest/seeds/. Those are
only ever read into an *empty* table, so they seed a new database without
ever overwriting one that's already been curated.

Every step is skip-if-already-done, so this is safe to run on every boot:
a warm deployment does one query and exits. That is what makes it usable as
an init container rather than a one-off — it must stay cheap and idempotent,
so think twice before adding a step that isn't both.

What it deliberately does *not* do is refresh anything already present.
Re-scraping a third-party site and re-embedding ~1,300 documents on every
container restart would be both wasteful and rude; keeping data current
stays a deliberate `ingest`-profile run (see docs/data-pipeline.md).

Usage:
    python -m riftbound_bot.ingest.bootstrap
"""
from __future__ import annotations

import structlog

from riftbound_bot.config import Settings
from riftbound_bot.ingest import seeds
from riftbound_bot.ingest.build_index import build_index
from riftbound_bot.ingest.cards_scrape import scrape_all_cards
from riftbound_bot.ingest.db import (
    CARDS_TABLE,
    RULES_TABLE,
    get_connection,
    update_card_images,
    upsert_cards,
)
from riftbound_bot.ingest.rules_import import import_chunks
from riftbound_bot.ingest.rules_parser import parse_rules_text
from riftbound_bot.logging_config import configure_logging
from riftbound_bot.rag.vectorstore import index_populated

logger = structlog.get_logger("riftbound_bot.bootstrap")


def _row_count(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        return cur.fetchone()[0]


def _ensure_rules(settings) -> None:
    """Seeds the hand translation only when the table is empty.

    Postgres is the source of truth for rules, so this must never overwrite
    what's there: the seed is a starting point for a fresh database, not a
    file the deployment re-syncs from. Re-importing on every boot would
    silently revert rules edited in the database — and prune any added
    since the shipped snapshot.
    """
    with get_connection(settings) as conn:
        existing = _row_count(conn, RULES_TABLE)
    if existing:
        logger.info("bootstrap.rules_present", count=existing)
        return

    chunks = parse_rules_text(seeds.rules_markdown(), source_file=seeds.RULES_SEED)
    import_chunks(settings, chunks)
    logger.info("bootstrap.rules_seeded", count=len(chunks))


def _ensure_cards(settings) -> None:
    """Scrapes cards only when the table is empty, falling back to the seed.

    The scrape is the one step here that reaches a third-party site, and
    the one most likely to break (it reads chroniclecore.com's internal
    /gallery payload). Gating it on an empty table means the common
    recovery case — embeddings dropped, `cards` intact — rebuilds from the
    cards already stored instead of depending on that site being up and
    unchanged.

    When it does break, the shipped snapshot takes over so that a fresh
    deployment still comes up. That deliberately does not extend to
    cards_from_api: falling back to a file costs nothing, while that path
    translates every card through the generation model — real spend the
    operator hasn't opted into by running `compose up`.
    """
    with get_connection(settings) as conn:
        existing = _row_count(conn, CARDS_TABLE)
        if existing:
            logger.info("bootstrap.cards_present", count=existing)
            return

        records, source = _scrape_or_seed(reason="cards_table_empty")

        if not records:
            raise RuntimeError(
                "Card scrape failed and the bundled snapshot is empty, so there "
                "is no card data to index. To pull fresh data through the "
                "Riftcodex fallback (this costs real LLM calls, which is why "
                "bootstrap will not start it for you):\n"
                "  docker compose --profile tools run --rm ingest "
                "riftbound_bot.ingest.cards_from_api\n"
                "See docs/data-pipeline.md."
            )

        upsert_cards(conn, records)
        logger.info("bootstrap.cards_loaded", count=len(records), source=source)


def _scrape_or_seed(reason: str) -> tuple[list[dict], str]:
    """Fresh card records from the site, falling back to the bundled snapshot.

    Shared by the two callers below because the failure argument is the same
    in both: the scrape reads a third-party site's internal payload and can
    fail as an HTTP error, a JSON error, or a KeyError on a renamed field.
    Enumerating those would just mean the one unlisted failure mode takes down
    the deployment, which is what the seed fallback exists to prevent.
    """
    logger.info("bootstrap.cards_scrape_start", reason=reason)
    try:
        return [card.__dict__ for card in scrape_all_cards()], "scrape"
    except Exception as error:  # noqa: BLE001
        logger.warning("bootstrap.cards_scrape_failed", error=str(error))
        return seeds.cards(), "seed"


def _ensure_card_images(settings) -> None:
    """Backfills card images into rows stored before the scraper captured them.

    This is the one step that refreshes data already present, which the module
    docstring otherwise rules out — so it is worth saying why. /card renders a
    card face, and card rows written before the scraper captured `assets` have
    none, so without this the deploy that adds /card would leave every existing
    deployment showing card embeds with a hole in them until an operator ran
    the scrape by hand.

    It writes *only* `image_url`, via update_card_images rather than
    upsert_cards. A re-scrape also brings back renames and recategorizations,
    and those feed the embedded text — writing them here, on the boot path that
    then skips the index rebuild, would silently leave retrieval matching names
    the cards table no longer holds. Keeping data current stays a deliberate
    refresh; this repairs one field.

    It stays honest about bootstrap's skip-if-already-done contract: the check
    is a single count that returns zero forever after the first successful
    backfill, so a warm deployment still does one query and exits.
    """
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) FROM {CARDS_TABLE} "
                "WHERE coalesce(data->>'image_url', '') = ''"
            )
            missing = cur.fetchone()[0]
        if not missing:
            return

        records, source = _scrape_or_seed(reason="cards_missing_images")
        images = {
            record["id"]: record.get("image_url", "")
            for record in records
            if record.get("id")
        }
        if not any(images.values()):
            raise RuntimeError(
                f"{missing} stored cards have no image, and neither the scrape "
                "nor the bundled snapshot produced any. See docs/data-pipeline.md."
            )
        updated = update_card_images(conn, images)
        logger.info(
            "bootstrap.card_images_backfilled",
            was_missing=missing,
            updated=updated,
            source=source,
        )


def bootstrap(settings) -> bool:
    """Returns True if it built the index, False if there was nothing to do."""
    # Ahead of the index check on purpose: this has to run on warm deployments
    # too, and those return early below.
    _ensure_card_images(settings)

    with get_connection(settings) as conn:
        if index_populated(conn):
            logger.info("bootstrap.index_present")
            return False

    _ensure_rules(settings)
    _ensure_cards(settings)

    count = build_index(settings)
    logger.info("bootstrap.index_built", documents=count)
    return True


def main() -> None:
    configure_logging()
    settings = Settings.load_for_ingest()
    if bootstrap(settings):
        print("Bootstrap complete. Index built in Postgres.")
    else:
        print("Index already present in Postgres; nothing to do.")


if __name__ == "__main__":
    main()
