"""CLI: writes the `rules` table back out as `[rule.id] title / body` Markdown.

Postgres holds the live copy of a hand translation that exists nowhere else,
so it needs a way out that isn't a database dump: this is the backup path, the
diffable review format, and how the shipped seed corpus is regenerated after
rules are edited (see ingest/seeds/).

Round-trips with rules_import — importing this output reproduces the same rows.

What it does *not* reproduce is any `<!-- -->` comment block in the original
file, because the parser strips comments before chunking and they never reach
Postgres. The seed corpus opens with exactly such a block documenting the
authoring convention, so regenerating that file from here means pasting the
header back on top; otherwise the next person to edit it loses the format
guide.

Usage:
    python -m riftbound_bot.ingest.rules_export > rules.md
    python -m riftbound_bot.ingest.rules_export -o src/riftbound_bot/ingest/seeds/core_rules_zh_tw.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from riftbound_bot.config import Settings
from riftbound_bot.ingest.db import RULES_TABLE, get_connection
from riftbound_bot.ingest.rules_parser import RuleChunk


def _sort_key(rule_id: str) -> list:
    """Orders rule ids by their numbering rather than as strings, so 9 comes
    before 10 and 805.2 before 805.10 — the order the source document uses."""
    parts = []
    for part in rule_id.split("."):
        parts.append((0, int(part), "") if part.isdigit() else (1, 0, part))
    return parts


def fetch_chunks(settings) -> list[RuleChunk]:
    with get_connection(settings) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT data FROM {RULES_TABLE}")
        chunks = [RuleChunk(**row[0]) for row in cur.fetchall()]
    return sorted(chunks, key=lambda chunk: _sort_key(chunk.rule_id))


def render(chunks: list[RuleChunk]) -> str:
    blocks = []
    for chunk in chunks:
        block = f"[{chunk.rule_id}] {chunk.title}".rstrip()
        if chunk.body:
            block += f"\n{chunk.body}"
        blocks.append(block)
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o", "--output", help="write here instead of stdout", default=None
    )
    args = parser.parse_args()

    settings = Settings.load_for_ingest()
    chunks = fetch_chunks(settings)
    if not chunks:
        sys.exit("No rules in Postgres — nothing to export.")

    markdown = render(chunks)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"Done. Exported {len(chunks)} rule chunks to {args.output}.")
    else:
        sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
