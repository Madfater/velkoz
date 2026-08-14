"""Tests for the first-run bootstrap that the `bootstrap` compose service
runs ahead of the bot.

The Postgres-backed cases follow test_db.py's probe-and-skip pattern; the
rest are pure and always run.
"""
import os

import psycopg
import pytest

from riftbound_bot.config import IngestSettings
from riftbound_bot.ingest import bootstrap as bootstrap_module
from riftbound_bot.ingest.db import (
    CARDS_TABLE,
    RULES_TABLE,
    get_connection,
    upsert_cards,
)
from riftbound_bot.rag.vectorstore import index_exists, load_vectorstore

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://riftbound:riftbound@localhost:5432/riftbound")


def _settings(vector_store_dir, rules_dir: str = "unused") -> IngestSettings:
    return IngestSettings(
        embedding_base_url="unused",
        embedding_api_key="unused",
        embedding_model="unused",
        vector_store_dir=str(vector_store_dir),
        rules_dir=rules_dir,
        database_url=DATABASE_URL,
    )


def _built_index(tmp_path):
    persist_dir = tmp_path / "turbovec"
    persist_dir.mkdir()
    (persist_dir / "docstore.json").write_text("{}", encoding="utf-8")
    return persist_dir


def test_index_exists_ignores_a_directory_with_no_docstore(tmp_path):
    persist_dir = tmp_path / "turbovec"
    persist_dir.mkdir()
    assert index_exists(str(persist_dir)) is False

    (persist_dir / "docstore.json").write_text("{}", encoding="utf-8")
    assert index_exists(str(persist_dir)) is True


def test_load_vectorstore_rejects_an_interrupted_build(tmp_path):
    """build_index creates persist_dir before dumping into it, so a crashed
    build leaves a bare directory behind. That has to keep raising the
    actionable "run bootstrap" error rather than failing on a missing
    docstore.json somewhere inside TurboVec."""
    persist_dir = tmp_path / "turbovec"
    persist_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="bootstrap"):
        load_vectorstore(str(persist_dir), embeddings=None)


def test_bootstrap_is_a_noop_when_the_index_is_already_built(tmp_path, monkeypatch):
    """The whole reason this can run as an init container on every `up`."""
    def fail(*args, **kwargs):
        raise AssertionError("a warm deployment must not re-run ingest")

    monkeypatch.setattr(bootstrap_module, "sync_rules", fail)
    monkeypatch.setattr(bootstrap_module, "_ensure_cards", fail)
    monkeypatch.setattr(bootstrap_module, "build_index", fail)

    assert bootstrap_module.bootstrap(_settings(_built_index(tmp_path))) is False


def test_bootstrap_populates_postgres_before_indexing_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap_module, "sync_rules", lambda s: calls.append("rules"))
    monkeypatch.setattr(bootstrap_module, "_ensure_cards", lambda s: calls.append("cards"))
    monkeypatch.setattr(
        bootstrap_module, "build_index", lambda s: calls.append("index") or 7
    )

    assert bootstrap_module.bootstrap(_settings(tmp_path / "missing")) is True
    # build_index reads both tables, so it has to come last.
    assert calls == ["rules", "cards", "index"]


@pytest.fixture
def conn():
    try:
        connection = get_connection(_settings("unused"))
    except psycopg.OperationalError:
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}")
    with connection.cursor() as cur:
        cur.execute(f"TRUNCATE {CARDS_TABLE}, {RULES_TABLE}")
    yield connection
    with connection.cursor() as cur:
        cur.execute(f"TRUNCATE {CARDS_TABLE}, {RULES_TABLE}")
    connection.close()


def test_ensure_cards_skips_the_scrape_when_cards_are_already_stored(conn, monkeypatch):
    """Index deleted but the Postgres volume intact — the common recovery
    case — must not depend on the scraped site being up."""
    upsert_cards(conn, [{"id": "C1", "name_zh": "卡一"}])

    def fail():
        raise AssertionError("should not reach the network when cards exist")

    monkeypatch.setattr(bootstrap_module, "scrape_all_cards", fail)
    bootstrap_module._ensure_cards(_settings("unused"))


def test_ensure_cards_scrapes_into_an_empty_table(conn, monkeypatch):
    class FakeCard:
        def __init__(self):
            self.__dict__ = {"id": "C9", "name_zh": "新卡"}

    monkeypatch.setattr(bootstrap_module, "scrape_all_cards", lambda: [FakeCard()])

    bootstrap_module._ensure_cards(_settings("unused"))

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
        bootstrap_module._ensure_cards(_settings("unused"))
