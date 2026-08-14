"""CLI: syncs data/rules/*.md into the `rules` Postgres table.

The Markdown files stay the hand-translated, git-tracked authoring source
(see README) — this script only loads what rules_parser.parse_rules_dir()
already parses from them into Postgres, so build_index.py has one consistent
data source (Postgres) for both rules and cards instead of a file-read for
one and a JSON-load for the other.

Usage:
    python -m riftbound_bot.ingest.rules_sync
"""
from __future__ import annotations

import structlog
from psycopg.types.json import Jsonb

from riftbound_bot.config import Settings
from riftbound_bot.ingest.db import RULES_TABLE, get_connection
from riftbound_bot.ingest.rules_parser import parse_rules_dir
from riftbound_bot.logging_config import configure_logging

logger = structlog.get_logger("riftbound_bot.rules_sync")

_UPSERT_SQL = f"""
INSERT INTO {RULES_TABLE} (rule_id, data)
VALUES (%s, %s)
ON CONFLICT (rule_id) DO UPDATE SET data = EXCLUDED.data
"""


def sync_rules(settings) -> int:
    chunks = parse_rules_dir(settings.rules_dir)
    if not chunks:
        # Load-bearing beyond the error message: the prune below deletes every
        # row whose id isn't in `fresh_ids`, and `!= ALL('{}')` is true for
        # every row — an empty parse would wipe the table rather than no-op.
        raise RuntimeError(f"No rule chunks found under {settings.rules_dir}")

    rows = [
        (
            chunk.rule_id,
            Jsonb(
                {
                    "rule_id": chunk.rule_id,
                    "title": chunk.title,
                    "body": chunk.body,
                    "source_file": chunk.source_file,
                }
            ),
        )
        for chunk in chunks
    ]
    fresh_ids = [rule_id for rule_id, _data in rows]

    # One transaction: upsert-then-prune are halves of a single sync, and
    # under autocommit an interruption between them left the table holding
    # the new rules plus whatever stale ones the prune hadn't reached yet.
    # psycopg commits on clean block exit and rolls back on exception.
    with get_connection(settings, autocommit=False) as conn, conn.cursor() as cur:
        cur.executemany(_UPSERT_SQL, rows)
        cur.execute(f"DELETE FROM {RULES_TABLE} WHERE rule_id != ALL(%s)", (fresh_ids,))
        pruned = cur.rowcount

    logger.info("rules_sync.done", upserted=len(rows), pruned=pruned)
    return len(rows)


def main() -> None:
    configure_logging()
    settings = Settings.load_for_ingest()
    count = sync_rules(settings)
    print(f"Done. Synced {count} rule chunks into Postgres.")


if __name__ == "__main__":
    main()
