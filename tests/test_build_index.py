"""Tests for the Postgres-backed index build.

Follows test_db.py's probe-and-skip pattern — these exercise real pgvector
DDL and distance queries, which a fake connection can't stand in for.
"""
import os

import psycopg
import pytest
from psycopg.types.json import Jsonb

from riftbound_bot.config import IngestSettings
from riftbound_bot.ingest import build_index as build_index_module
from riftbound_bot.ingest.db import (
    CARDS_TABLE,
    EMBEDDINGS_TABLE,
    RULES_TABLE,
    embedding_dimension,
    get_connection,
    upsert_cards,
)
from riftbound_bot.rag.vectorstore import PgVectorStore

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://riftbound:riftbound@localhost:5432/riftbound")


def _settings() -> IngestSettings:
    return IngestSettings(
        embedding_base_url="unused",
        embedding_api_key="unused",
        embedding_model="unused",
        rules_dir="unused",
        database_url=DATABASE_URL,
    )


class FakeEmbeddings:
    """Returns a fixed-width vector per document, so distances are predictable.

    `vectors` maps a substring of the document text to the vector it embeds to;
    anything unmatched gets `default`.
    """

    def __init__(self, vectors: dict[str, list[float]], default: list[float]):
        self._vectors = vectors
        self._default = default
        self.batches: list[int] = []

    def _embed(self, text: str) -> list[float]:
        for needle, vector in self._vectors.items():
            if needle in text:
                return vector
        return self._default

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


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
        cur.execute(f"DROP TABLE IF EXISTS {EMBEDDINGS_TABLE}_build")


def _seed_corpus(connection) -> None:
    with connection.cursor() as cur:
        cur.execute(
            f"INSERT INTO {RULES_TABLE} (rule_id, data) VALUES (%s, %s)",
            (
                "805.1",
                Jsonb(
                    {
                        "rule_id": "805.1",
                        "title": "疾行是一種單位能力。",
                        "body": "",
                        "source_file": "core.md",
                    }
                ),
            ),
        )
    upsert_cards(
        connection,
        [{"id": "OGN-001", "name_zh": "測試卡", "name_en": "Test", "category": "單位"}],
    )


def _build(connection, monkeypatch, embeddings) -> int:
    monkeypatch.setattr(build_index_module, "build_embeddings", lambda **kwargs: embeddings)
    return build_index_module.build_index(_settings())


def test_build_index_sizes_the_table_from_the_returned_vectors(conn, monkeypatch):
    """Nothing configures the embedding dimension, so the table has to take it
    from whatever the endpoint actually returns — otherwise a model swap would
    need a setting updated in lockstep to avoid a silently wrong `vector(N)`."""
    _seed_corpus(conn)

    assert _build(conn, monkeypatch, FakeEmbeddings({}, default=[0.1, 0.2, 0.3])) == 2

    assert embedding_dimension(conn) == 3


def test_build_index_resizes_when_the_embedding_model_changes(conn, monkeypatch):
    _seed_corpus(conn)
    _build(conn, monkeypatch, FakeEmbeddings({}, default=[0.1, 0.2, 0.3]))
    assert embedding_dimension(conn) == 3

    _build(conn, monkeypatch, FakeEmbeddings({}, default=[0.1, 0.2, 0.3, 0.4]))

    assert embedding_dimension(conn) == 4


def test_a_failed_rebuild_leaves_the_previous_index_queryable(conn, monkeypatch):
    """The whole reason the build writes to a side table and renames it last:
    a crash mid-rebuild must not leave the bot serving a half-populated index."""
    _seed_corpus(conn)
    _build(conn, monkeypatch, FakeEmbeddings({}, default=[1.0, 0.0]))

    class Exploding(FakeEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("embedding endpoint died mid-rebuild")

    with pytest.raises(RuntimeError, match="died mid-rebuild"):
        _build(conn, monkeypatch, Exploding({}, default=[0.0, 1.0]))

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {EMBEDDINGS_TABLE}")
        assert cur.fetchone()[0] == 2
    assert embedding_dimension(conn) == 2


def test_indexed_documents_round_trip_through_the_store(conn, monkeypatch):
    """End-to-end over the real SQL: build writes rows, the store reads back
    the same page_content/metadata and ranks by cosine distance."""
    _seed_corpus(conn)
    embeddings = FakeEmbeddings(
        {"疾行": [1.0, 0.0], "測試卡": [0.0, 1.0]}, default=[0.0, 0.0]
    )
    _build(conn, monkeypatch, embeddings)

    store = PgVectorStore(_PoolOf(conn), embeddings)

    rules = store.fetch_by_source_type("rule")
    assert [doc.metadata["rule_id"] for doc in rules] == ["805.1"]
    assert "疾行" in rules[0].page_content

    # A query embedding identical to the rule's vector is distance 0, which the
    # shared (similarity + 1) / 2 mapping turns into a relevance of 1.0.
    hits = store.similarity_search_with_relevance_scores("疾行", k=5, filter={"source_type": "rule"})
    assert [doc.metadata["rule_id"] for doc, _ in hits] == ["805.1"]
    assert hits[0][1] == pytest.approx(1.0)

    # The filter is a hard partition: a rule query never returns cards.
    cards = store.similarity_search("疾行", k=5, filter={"source_type": "card"})
    assert [doc.metadata["card_id"] for doc in cards] == ["OGN-001"]


class _PoolOf:
    """Minimal stand-in for ConnectionPool.connection() over one open handle,
    so the store can be exercised without a second pool against the same test
    database."""

    def __init__(self, connection):
        self._connection = connection

    def connection(self):
        pool_connection = self._connection

        class _Ctx:
            def __enter__(self):
                return pool_connection

            def __exit__(self, *exc_info):
                return False

        return _Ctx()
