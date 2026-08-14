"""Integration tests against a real (ephemeral) Postgres — no mock captures
SQL semantics like ON CONFLICT upserts or JSONB round-tripping faithfully
enough to trust. Skipped when Postgres isn't actually reachable (e.g. plain
`pytest` locally without it running); CI provides a postgres service
container.

Deliberately probes with a real (short-timeout) connection attempt rather
than checking whether the DATABASE_URL env var is merely *set* —
config.py's load_dotenv() is a process-wide side effect of importing
riftbound_bot.config at all (triggered by other test modules collected
first, e.g. test_bot.py), so DATABASE_URL can be populated from .env's
local-dev default even when nothing is actually listening there.
"""
import os

import psycopg
import pytest

from riftbound_bot.config import IngestSettings
from riftbound_bot.ingest.db import CARDS_TABLE, RULES_TABLE, get_connection

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://riftbound:riftbound@localhost:5432/riftbound")


def _settings(rules_dir: str = "unused") -> IngestSettings:
    return IngestSettings(
        embedding_base_url="unused",
        embedding_api_key="unused",
        embedding_model="unused",
        rules_dir=rules_dir,
        database_url=DATABASE_URL,
    )


@pytest.fixture
def conn():
    try:
        connection = get_connection(_settings())
    except psycopg.OperationalError:
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}")
    with connection.cursor() as cur:
        cur.execute(f"TRUNCATE {CARDS_TABLE}, {RULES_TABLE}")
    yield connection
    with connection.cursor() as cur:
        cur.execute(f"TRUNCATE {CARDS_TABLE}, {RULES_TABLE}")
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
