"""Integration tests against a real (scratch) Postgres — no mock captures
SQL semantics like ON CONFLICT upserts or JSONB round-tripping faithfully
enough to trust.

These tests TRUNCATE the `cards` and `rules` tables, so they deliberately
read TEST_DATABASE_URL and *never* fall back to DATABASE_URL: the latter is
populated process-wide from .env by config.py's import-time load_dotenv()
and normally points at the developer's real, fully-ingested local database.
Falling back to it would make a plain `pytest` destroy ~1,300 scraped cards
and the whole rules corpus.

Setting TEST_DATABASE_URL is therefore both the opt-in *and* an assertion
that the database must be reachable: unset skips, but set-and-unreachable
fails. That way a dead CI postgres service turns the build red instead of
silently skipping the only tests covering real SQL.
"""
import os

import psycopg
import pytest

from riftbound_bot.config import IngestSettings
from riftbound_bot.ingest.db import CARDS_TABLE, RULES_TABLE, get_connection

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


def _settings(rules_dir: str = "unused") -> IngestSettings:
    return IngestSettings(
        embedding_base_url="unused",
        embedding_api_key="unused",
        embedding_model="unused",
        vector_store_dir="unused",
        rules_dir=rules_dir,
        database_url=TEST_DATABASE_URL or "",
    )


def _truncate(connection) -> None:
    with connection.cursor() as cur:
        cur.execute(f"TRUNCATE {CARDS_TABLE}, {RULES_TABLE}")


@pytest.fixture
def conn():
    if not TEST_DATABASE_URL:
        pytest.skip(
            "set TEST_DATABASE_URL to a scratch database to run the Postgres "
            "integration tests — its cards/rules tables get TRUNCATEd"
        )
    try:
        connection = get_connection(_settings())
    except psycopg.OperationalError as error:
        # Set-but-unreachable is a failure, not a skip: otherwise a CI postgres
        # service that never came up would leave the build green with zero
        # coverage of the only tests that exercise real SQL.
        pytest.fail(f"TEST_DATABASE_URL is set but Postgres is unreachable: {error}")

    try:
        _truncate(connection)
        yield connection
        _truncate(connection)
    finally:
        connection.close()


def test_upsert_cards_inserts_then_updates_in_place(conn):
    from riftbound_bot.ingest.db import upsert_cards

    upsert_cards(conn, [{"id": "C1", "name_zh": "卡一", "rarity": "普通"}])
    upsert_cards(conn, [{"id": "C1", "name_zh": "卡一", "rarity": "史詩"}])

    with conn.cursor() as cur:
        cur.execute("SELECT data FROM cards")
        rows = [row[0] for row in cur.fetchall()]

    assert rows == [{"id": "C1", "name_zh": "卡一", "rarity": "史詩"}]


def test_rules_sync_upserts_and_prunes_stale_rows(conn, tmp_path):
    from riftbound_bot.ingest.rules_sync import sync_rules

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "core.md").write_text("[1] 第一條規則。\n[2] 第二條規則。\n", encoding="utf-8")

    settings = _settings(rules_dir=str(rules_dir))
    sync_rules(settings)

    with conn.cursor() as cur:
        cur.execute("SELECT rule_id FROM rules ORDER BY rule_id")
        assert [row[0] for row in cur.fetchall()] == ["1", "2"]

    # Rule 2 removed from the source file — a re-sync should prune it.
    (rules_dir / "core.md").write_text("[1] 第一條規則（已修改）。\n", encoding="utf-8")
    sync_rules(settings)

    with conn.cursor() as cur:
        cur.execute("SELECT rule_id, data->>'title' FROM rules")
        assert cur.fetchall() == [("1", "第一條規則（已修改）。")]
