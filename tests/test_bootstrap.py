"""Tests for the first-run bootstrap that the `bootstrap` compose service
runs ahead of the bot.

All of these need Postgres now that the index lives there too, so they all
follow test_db.py's probe-and-skip pattern.
"""
import os

import psycopg
import pytest

from riftbound_bot.config import IngestSettings
from riftbound_bot.ingest import bootstrap as bootstrap_module
from riftbound_bot.ingest.db import (
    CARDS_TABLE,
    EMBEDDINGS_TABLE,
    RULES_TABLE,
    create_embeddings_build_table,
    get_connection,
    promote_embeddings_build_table,
    upsert_cards,
)
from riftbound_bot.rag.vectorstore import index_populated

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://riftbound:riftbound@localhost:5432/riftbound")


def _settings() -> IngestSettings:
    return IngestSettings(
        embedding_base_url="unused",
        embedding_api_key="unused",
        embedding_model="unused",
        rules_dir="unused",
        database_url=DATABASE_URL,
    )


@pytest.fixture
def conn():
    try:
        connection = get_connection(_settings())
    except psycopg.OperationalError:
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}")
    _reset(connection)
    yield connection
    _reset(connection)
    connection.close()


def _reset(connection) -> None:
    with connection.cursor() as cur:
        cur.execute(f"TRUNCATE {CARDS_TABLE}, {RULES_TABLE}")
        cur.execute(f"DROP TABLE IF EXISTS {EMBEDDINGS_TABLE}")


def _build_index_of(connection, rows: list[tuple[str, str]]) -> None:
    """Puts a real (tiny) index in place, via the same promote path build_index
    uses — so these tests exercise the actual swap rather than a stub table."""
    create_embeddings_build_table(connection, dim=2)
    with connection.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {EMBEDDINGS_TABLE}_build (id, source_type, data, embedding) "
            "VALUES (%s, %s, %s, %s::vector)",
            [
                (doc_id, source_type, '{"page_content": "x", "metadata": {}}', "[1,0]")
                for doc_id, source_type in rows
            ],
        )
    promote_embeddings_build_table(connection)


def test_index_populated_is_false_until_the_index_is_promoted(conn):
    """build_index fills a build table and renames it over the live one, so
    "the table exists" alone would report a half-built index as ready — and
    tell bootstrap to skip the very rebuild it exists to perform."""
    assert index_populated(conn) is False

    create_embeddings_build_table(conn, dim=2)
    assert index_populated(conn) is False

    _build_index_of(conn, [("rule:100", "rule")])
    assert index_populated(conn) is True


def test_index_populated_is_false_for_an_empty_table(conn):
    _build_index_of(conn, [])
    assert index_populated(conn) is False


def test_bootstrap_is_a_noop_when_the_index_is_already_built(conn, monkeypatch):
    """The whole reason this can run as an init container on every `up`."""
    _build_index_of(conn, [("rule:100", "rule")])

    def fail(*args, **kwargs):
        raise AssertionError("a warm deployment must not re-run ingest")

    monkeypatch.setattr(bootstrap_module, "import_rules", fail)
    monkeypatch.setattr(bootstrap_module, "_ensure_cards", fail)
    monkeypatch.setattr(bootstrap_module, "build_index", fail)

    assert bootstrap_module.bootstrap(_settings()) is False


def test_bootstrap_populates_postgres_before_indexing_it(conn, monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap_module, "import_rules", lambda s, src: calls.append("rules"))
    monkeypatch.setattr(bootstrap_module, "_ensure_cards", lambda s: calls.append("cards"))
    monkeypatch.setattr(
        bootstrap_module, "build_index", lambda s: calls.append("index") or 7
    )

    assert bootstrap_module.bootstrap(_settings()) is True
    # build_index reads both tables, so it has to come last.
    assert calls == ["rules", "cards", "index"]


def test_ensure_cards_skips_the_scrape_when_cards_are_already_stored(conn, monkeypatch):
    """Index deleted but the Postgres volume intact — the common recovery
    case — must not depend on the scraped site being up."""
    upsert_cards(conn, [{"id": "C1", "name_zh": "卡一"}])

    def fail():
        raise AssertionError("should not reach the network when cards exist")

    monkeypatch.setattr(bootstrap_module, "scrape_all_cards", fail)
    bootstrap_module._ensure_cards(_settings())


def test_ensure_cards_scrapes_into_an_empty_table(conn, monkeypatch):
    class FakeCard:
        def __init__(self):
            self.__dict__ = {"id": "C9", "name_zh": "新卡"}

    monkeypatch.setattr(bootstrap_module, "scrape_all_cards", lambda: [FakeCard()])

    bootstrap_module._ensure_cards(_settings())

    with conn.cursor() as cur:
        cur.execute("SELECT data FROM cards")
        assert [row[0] for row in cur.fetchall()] == [{"id": "C9", "name_zh": "新卡"}]


def test_ensure_cards_points_at_the_paid_fallback_when_the_scrape_breaks(conn, monkeypatch):
    """cards_from_api costs real LLM calls, so bootstrap names it instead of
    silently running it."""
    def boom():
        raise ValueError("gallery payload changed shape")

    monkeypatch.setattr(bootstrap_module, "scrape_all_cards", boom)

    with pytest.raises(RuntimeError, match="cards_from_api"):
        bootstrap_module._ensure_cards(_settings())
