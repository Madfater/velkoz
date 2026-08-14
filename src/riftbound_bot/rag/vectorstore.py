from __future__ import annotations

from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from turbovec.langchain import TurboQuantVectorStore

# TurboVec's compression story targets millions of vectors — irrelevant at
# this corpus's scale (~1,300 vectors total). bit_width=4 prioritizes recall
# over compression, which is the right tradeoff here. Cosine similarity is
# TurboQuantVectorStore's unconditional behavior in the installed release
# (0.8.0) — there's no similarity= kwarg to set (a cosine/dot_product toggle
# exists on turbovec's unreleased main branch; re-check this if the pin is
# ever bumped past 0.8.x).
BIT_WIDTH = 4

# .dump() writes this side-car next to the index file, and .load() opens it
# first — so its presence, not the directory's, is what "there is an index
# here" actually means.
DOCSTORE_FILENAME = "docstore.json"


def index_exists(persist_dir: str) -> bool:
    """Whether persist_dir holds a *complete* store rather than just a directory.

    build_index creates persist_dir before writing into it, so an
    interrupted build can leave the directory behind with no docstore in
    it. Reading a bare directory as "index present" is the wrong answer for
    both callers: ingest/bootstrap.py would skip the very rebuild it exists
    to perform, and the bot would then fail on a missing docstore.json deep
    inside TurboVec instead of the actionable error below.
    """
    return (Path(persist_dir) / DOCSTORE_FILENAME).is_file()


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


def create_vectorstore(embeddings: OpenAIEmbeddings) -> TurboQuantVectorStore:
    """Fresh, empty store for build_index.py to populate via add_documents()
    and then persist explicitly with .dump() — TurboVec's persistence is
    push-based, unlike Chroma's auto-persist-per-write.
    """
    return TurboQuantVectorStore(embedding=embeddings, bit_width=BIT_WIDTH)


def load_vectorstore(persist_dir: str, embeddings: OpenAIEmbeddings) -> TurboQuantVectorStore:
    """Loads a store previously written by create_vectorstore + .dump().

    Deliberately does not fall back to creating an empty store when
    persist_dir is missing/empty — a typo'd VECTOR_STORE_DIR should raise
    loudly ("index not found, run build_index"), not silently serve an
    empty store. This is a real reliability improvement over Chroma's old
    transparent create-or-load behavior, not just parity.
    """
    if not index_exists(persist_dir):
        raise FileNotFoundError(
            f"No vector store at {persist_dir} — run `python -m "
            "riftbound_bot.ingest.bootstrap` first."
        )
    return TurboQuantVectorStore.load(persist_dir, embedding=embeddings)
