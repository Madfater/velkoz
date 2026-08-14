"""Postgres access shared by the ingest CLIs (cards_scrape, cards_from_api,
rules_sync, build_index). Ingest-time only — the live bot never imports this
module; it reads the already-built vector store instead (see
rag/vectorstore.py's load_vectorstore).

Schema is a narrow-plus-JSONB shape rather than a fully normalized relational
one: a primary-key column to upsert against, and a `data` JSONB column
holding the whole record. This keeps the flexible, easy-upsert, JSON-shaped
storage the card/rule data already has as plain dicts, without hand-porting
scraped fields into rigid typed columns.
"""
from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from riftbound_bot.config import IngestSettings

CARDS_TABLE = "cards"
RULES_TABLE = "rules"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {CARDS_TABLE} (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS {RULES_TABLE} (
    rule_id TEXT PRIMARY KEY,
    data JSONB NOT NULL
);
"""

_UPSERT_CARD_SQL = f"""
INSERT INTO {CARDS_TABLE} (id, data)
VALUES (%s, %s)
ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
"""


def get_connection(settings: IngestSettings) -> psycopg.Connection:
    # Explicit connect_timeout: a dead/unreachable database should fail fast
    # with a clear error, not hang the ingest script indefinitely.
    conn = psycopg.connect(settings.database_url, autocommit=True, connect_timeout=3)
    ensure_schema(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA)


def upsert_cards(conn: psycopg.Connection, cards: list[dict]) -> None:
    """Upserts by `id` — a re-scrape updates existing cards in place instead
    of only ever replacing the whole dataset (the old cards.json behavior)."""
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_CARD_SQL, [(card["id"], Jsonb(card)) for card in cards])
