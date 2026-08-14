"""CLI: loads a `[rule.id] title / body` Markdown corpus into the `rules` table.

Postgres is the live copy of the hand translation — the bot's index is built
from these rows, not from any file. Markdown stays the editing format because
it is the format the translation is transcribed in (see docs/data-pipeline.md),
but it is an import/export interchange here rather than a directory the
deployment reads at runtime.

Takes an explicit path rather than a configured rules directory: there is no
tracked data/ folder to point at any more, so the source is wherever the
person doing the transcription happens to keep it.

Usage:
    python -m riftbound_bot.ingest.rules_import path/to/rules.md
    python -m riftbound_bot.ingest.rules_import path/to/rules_dir/
    python -m riftbound_bot.ingest.rules_import --no-prune path/to/extra.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import structlog
from psycopg.types.json import Jsonb

from riftbound_bot.config import Settings
from riftbound_bot.ingest.db import RULES_TABLE, get_connection
from riftbound_bot.ingest.rules_parser import (
    RuleChunk,
    parse_rules_dir,
    parse_rules_text,
)
from riftbound_bot.logging_config import configure_logging

logger = structlog.get_logger("riftbound_bot.rules_import")

_UPSERT_SQL = f"""
INSERT INTO {RULES_TABLE} (rule_id, data)
VALUES (%s, %s)
ON CONFLICT (rule_id) DO UPDATE SET data = EXCLUDED.data
"""


def parse_source(source: str | Path) -> list[RuleChunk]:
    """Reads either a single Markdown file or a directory of them."""
    source = Path(source)
    if source.is_dir():
        return parse_rules_dir(source)
    return parse_rules_text(source.read_text(encoding="utf-8"), source_file=source.name)


def import_chunks(settings, chunks: list[RuleChunk], prune: bool = True) -> int:
    """Upserts `chunks` by rule_id.

    `prune` deletes rows the import didn't mention, which is what keeps a
    full-corpus import an exact mirror of its source — a rule deleted from the
    Markdown disappears from the table too. It has to be off for a partial
    import, which would otherwise wipe every rule outside the file being added.
    """
    if not chunks:
        raise RuntimeError("No rule chunks parsed — nothing to import.")

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

    with get_connection(settings) as conn, conn.cursor() as cur:
        cur.executemany(_UPSERT_SQL, rows)
        pruned = 0
        if prune:
            fresh_ids = [rule_id for rule_id, _data in rows]
            cur.execute(f"DELETE FROM {RULES_TABLE} WHERE rule_id != ALL(%s)", (fresh_ids,))
            pruned = cur.rowcount

    logger.info("rules_import.done", upserted=len(rows), pruned=pruned)
    return len(rows)


def import_rules(settings, source: str | Path, prune: bool = True) -> int:
    return import_chunks(settings, parse_source(source), prune=prune)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="Markdown file, or a directory of *.md files")
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="keep rules absent from the source (use when importing a partial corpus)",
    )
    args = parser.parse_args()

    configure_logging()
    settings = Settings.load_for_ingest()
    count = import_rules(settings, args.source, prune=not args.no_prune)
    print(f"Done. Imported {count} rule chunks into Postgres.")


if __name__ == "__main__":
    main()
