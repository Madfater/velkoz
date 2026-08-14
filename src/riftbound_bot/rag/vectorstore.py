from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from psycopg_pool import ConnectionPool

from riftbound_bot.ingest.db import EMBEDDINGS_TABLE


class RetrievalStore(Protocol):
    """The entire vector-store surface rag/chain.py depends on.

    Spelled out as a Protocol rather than reusing langchain_core's VectorStore
    because that base class advertises dozens of methods this project never
    calls, which made the real dependency — three calls — impossible to see.
    Structural typing also keeps the tests' fake stores valid without
    inheriting anything.
    """

    def similarity_search_with_relevance_scores(
        self, query: str, k: int, filter: dict
    ) -> list[tuple[Document, float]]: ...

    def similarity_search(self, query: str, k: int, filter: dict) -> list[Document]: ...

    def fetch_by_source_type(self, source_type: str) -> list[Document]: ...


def build_embeddings(base_url: str, api_key: str, model: str) -> OpenAIEmbeddings:
    """Embeddings via a self-hosted, OpenAI-compatible /v1/embeddings endpoint
    (not a commercial provider) — zero marginal API cost, own infrastructure.

    Explicit timeout/max_retries: the SDK's own defaults would otherwise let
    a stalled self-hosted endpoint hang far longer than a Discord interaction
    should ever wait.
    """
    return OpenAIEmbeddings(
        base_url=base_url, api_key=api_key, model=model, timeout=30, max_retries=3
    )


def to_vector_literal(values: list[float]) -> str:
    """Formats an embedding as pgvector's text input form, e.g. '[0.1,0.2]'.

    Passed as a plain string parameter with an explicit `::vector` cast at
    every call site, which is why this project doesn't depend on the
    `pgvector` Python package: nothing ever reads an embedding column *back*
    out of Postgres — queries return the payload and a computed distance — so
    a result-side type adapter would buy nothing.
    """
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _document(payload: dict) -> Document:
    return Document(page_content=payload["page_content"], metadata=payload["metadata"])


def _relevance(distance: float) -> float:
    """Maps pgvector cosine distance to the 0-1 relevance scale.

    `<=>` returns 1 - cosine_similarity, so similarity is 1 - distance and
    this is (similarity + 1) / 2 — deliberately the same mapping TurboVec
    applied, so RETRIEVAL_SCORE_THRESHOLD keeps the meaning it was tuned
    with and swapping the vector store doesn't silently move the cutoff.
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


class PgVectorStore:
    """Read-only view of the `embeddings` table for the running bot.

    Exposes just the three calls rag/chain.py makes, not a general vector-store
    abstraction: the corpus is ~1,300 rows, so every query is an exact
    sequential scan with `<=>` rather than an ANN lookup. That is both fast
    enough (sub-millisecond) and strictly better recall than an approximate
    index — worth revisiting only somewhere north of ~100k rows.

    Queries go through a pool rather than one long-lived connection because
    the bot stays up for days at a time; a connection dropped by the server or
    a network blip would otherwise take the bot down until it was restarted.
    """

    def __init__(self, pool: ConnectionPool, embeddings: OpenAIEmbeddings):
        self._pool = pool
        self._embeddings = embeddings

    def similarity_search_with_relevance_scores(
        self, query: str, k: int, filter: dict
    ) -> list[tuple[Document, float]]:
        vector = to_vector_literal(self._embeddings.embed_query(query))
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT data, embedding <=> %s::vector AS distance
                FROM {EMBEDDINGS_TABLE}
                WHERE source_type = %s
                ORDER BY distance
                LIMIT %s
                """,
                (vector, filter["source_type"], k),
            )
            rows = cur.fetchall()
        return [(_document(payload), _relevance(distance)) for payload, distance in rows]

    def similarity_search(self, query: str, k: int, filter: dict) -> list[Document]:
        results = self.similarity_search_with_relevance_scores(query, k, filter)
        return [document for document, _score in results]

    def fetch_by_source_type(self, source_type: str) -> list[Document]:
        """Every document of one type, with no query vector involved.

        chain.py's exact-match indexes need the complete set, not a ranking.
        Under TurboVec that had to be faked with a throwaway empty query and a
        huge k, which cost an embedding API call per source type on every boot;
        in SQL it's just a SELECT.
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT data FROM {EMBEDDINGS_TABLE} WHERE source_type = %s",
                (source_type,),
            )
            return [_document(row[0]) for row in cur.fetchall()]


def index_populated(conn) -> bool:
    """Whether the embeddings table exists and has rows in it.

    Both halves matter, and for the same reason the old on-disk check looked
    for the docstore file rather than its directory: build_index creates the
    table before filling it, so "exists" alone would report a half-built index
    as ready — telling bootstrap to skip the rebuild it exists to perform.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (EMBEDDINGS_TABLE,))
        if cur.fetchone()[0] is None:
            return False
        cur.execute(f"SELECT EXISTS (SELECT 1 FROM {EMBEDDINGS_TABLE})")
        return cur.fetchone()[0]
