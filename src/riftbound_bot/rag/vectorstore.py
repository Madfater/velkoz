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
    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"No vector store at {persist_dir} — run `python -m "
            "riftbound_bot.ingest.build_index` first."
        )
    return TurboQuantVectorStore.load(persist_dir, embedding=embeddings)
