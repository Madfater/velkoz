"""Postgres schema and connections, shared by the ingest CLIs (cards_scrape,
cards_from_api, rules_import, build_index) and the live bot.

Three tables. `cards` and `rules` are ingest-time sources; `embeddings` is the
vector index the bot actually serves from, and is the one table read at
request time. Writes are still ingest-only — the bot never writes here.

Schema is a narrow-plus-JSONB shape rather than a fully normalized relational
one: the columns worth querying or upserting against, and a `data` JSONB
column holding the whole record. This keeps the flexible, easy-upsert,
JSON-shaped storage the card/rule data already has as plain dicts, without
hand-porting scraped fields into rigid typed columns.
"""
from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from riftbound_bot.config import IngestSettings

CARDS_TABLE = "cards"
RULES_TABLE = "rules"
EMBEDDINGS_TABLE = "embeddings"

# build_index writes here first and renames it over EMBEDDINGS_TABLE at the
# very end, so a crash mid-build leaves the previous index queryable.
EMBEDDINGS_BUILD_TABLE = f"{EMBEDDINGS_TABLE}_build"

_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {CARDS_TABLE} (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS {RULES_TABLE} (
    rule_id TEXT PRIMARY KEY,
    data JSONB NOT NULL
);
"""

# Not part of _SCHEMA: pgvector's `vector(N)` needs its dimension up front, and
# that comes from whatever the embedding endpoint returns — see
# create_embeddings_build_table. `source_type` is a real column rather than a
# JSONB expression because it's the only field ever filtered on; `data` keeps
# the narrow-plus-JSONB shape the other two tables use.
_EMBEDDINGS_DDL = """
CREATE TABLE {table} (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    data JSONB NOT NULL,
    embedding vector({dim}) NOT NULL
)
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


def update_card_images(conn: psycopg.Connection, images: dict[str, str]) -> int:
    """Sets `image_url` on stored cards, touching no other field.

    Deliberately not `upsert_cards`: a re-scrape brings back renames,
    recategorizations and retranslated tags along with the images, and all of
    those feed `CardRecord.text` — the string that gets embedded. Writing them
    without also rebuilding the index would leave retrieval matching against
    names the `cards` table no longer holds. A backfill should repair the one
    field it is named for and leave the rest for a deliberate refresh.

    Only fills the gap, never overwrites an image already stored, so re-running
    it cannot revert a hand-corrected URL.
    """
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            UPDATE {CARDS_TABLE}
            -- Explicit ::text because jsonb_build_object is variadic "any",
            -- so Postgres cannot infer a bare parameter's type here.
            SET data = data || jsonb_build_object('image_url', %s::text)
            WHERE id = %s AND coalesce(data->>'image_url', '') = ''
            """,
            [(url, card_id) for card_id, url in images.items() if url],
        )
        return cur.rowcount


def embedding_dimension(conn: psycopg.Connection, table: str = EMBEDDINGS_TABLE) -> int | None:
    """Vector width of `table`'s embedding column, or None if it doesn't exist.

    pgvector stores a `vector(N)` column's N directly in `atttypmod` (unlike
    varchar, which offsets it), so this reads the declared dimension without
    having to parse format_type() output.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = to_regclass(%s) AND attname = 'embedding'
            """,
            (table,),
        )
        row = cur.fetchone()
    if row is None or row[0] is None or row[0] < 0:
        return None
    return row[0]


def create_embeddings_build_table(conn: psycopg.Connection, dim: int) -> None:
    """Fresh, empty build table sized to `dim`.

    Dropped-and-recreated rather than reused so a rebuild after an embedding
    model change can't inherit the old width — the dimension always comes from
    the vectors actually being written.
    """
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {EMBEDDINGS_BUILD_TABLE}")
        cur.execute(_EMBEDDINGS_DDL.format(table=EMBEDDINGS_BUILD_TABLE, dim=dim))


def promote_embeddings_build_table(conn: psycopg.Connection) -> None:
    """Swaps the finished build table over the live one, in one transaction.

    Everything before this point writes only to the build table, so an
    interrupted rebuild leaves the bot serving the previous index rather than
    a half-populated one.
    """
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {EMBEDDINGS_TABLE}")
        cur.execute(f"ALTER TABLE {EMBEDDINGS_BUILD_TABLE} RENAME TO {EMBEDDINGS_TABLE}")
        # Index names are schema-scoped, so the pkey has to be renamed too or
        # the next build's CREATE TABLE collides with it and gets an
        # auto-suffixed name.
        cur.execute(
            f"ALTER INDEX {EMBEDDINGS_BUILD_TABLE}_pkey RENAME TO {EMBEDDINGS_TABLE}_pkey"
        )
        cur.execute(
            f"CREATE INDEX {EMBEDDINGS_TABLE}_source_type_idx "
            f"ON {EMBEDDINGS_TABLE} (source_type)"
        )
