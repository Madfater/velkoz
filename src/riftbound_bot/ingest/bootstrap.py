"""CLI: brings a deployment up to a servable state, then exits.

`docker compose up` runs this before the bot starts (see the `bootstrap`
service in docker-compose.yml), so a clean checkout produces a working bot
instead of one that crash-loops on a missing index until three ingest
commands are run by hand in the right order.

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
from riftbound_bot.ingest.build_index import build_index
from riftbound_bot.ingest.cards_scrape import scrape_all_cards
from riftbound_bot.ingest.db import CARDS_TABLE, get_connection, upsert_cards
from riftbound_bot.ingest.rules_import import import_rules
from riftbound_bot.logging_config import configure_logging
from riftbound_bot.rag.vectorstore import index_populated

logger = structlog.get_logger("riftbound_bot.bootstrap")


def _ensure_cards(settings) -> None:
    """Scrapes cards only when the table is empty.

    The scrape is the one step here that reaches a third-party site, and
    the one most likely to break (it reads chroniclecore.com's internal
    /gallery payload). Gating it on an empty table means the common
    recovery case — embeddings dropped, `cards` intact — rebuilds from the
    cards already stored instead of depending on that site being up and
    unchanged.
    """
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {CARDS_TABLE}")
            existing = cur.fetchone()[0]
        if existing:
            logger.info("bootstrap.cards_present", count=existing)
            return

        logger.info("bootstrap.cards_scrape_start")
        try:
            cards = scrape_all_cards()
        except Exception as error:
            # Don't auto-fall back to cards_from_api: that path translates
            # every card through the generation model, which is real spend
            # the operator hasn't opted into by running `compose up`.
            raise RuntimeError(
                "Card scrape failed, so there is no card data to index. This "
                "step depends on chroniclecore.com's internal /gallery data "
                "shape and breaks when the site changes it. The documented "
                "fallback translates the Riftcodex API through the generation "
                "model — that costs real LLM calls, so bootstrap will not "
                "start it for you. To use it:\n"
                "  docker compose --profile tools run --rm ingest "
                "riftbound_bot.ingest.cards_from_api\n"
                "See docs/data-pipeline.md."
            ) from error

        upsert_cards(conn, [card.__dict__ for card in cards])
        logger.info("bootstrap.cards_scraped", count=len(cards))


def bootstrap(settings) -> bool:
    """Returns True if it built the index, False if there was nothing to do."""
    with get_connection(settings) as conn:
        if index_populated(conn):
            logger.info("bootstrap.index_present")
            return False

    # Rules come from git-tracked Markdown — offline, cheap, and the import
    # already upserts-and-prunes to match the source exactly, so it's simpler
    # to just always run it than to detect whether the table has drifted.
    import_rules(settings, settings.rules_dir)
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
