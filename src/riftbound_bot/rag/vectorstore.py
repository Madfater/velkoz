from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

COLLECTION_NAME = "riftbound"


def get_embeddings(base_url: str, api_key: str, model: str) -> OpenAIEmbeddings:
    """Embeddings via a self-hosted, OpenAI-compatible /v1/embeddings endpoint
    (not a commercial provider) — zero marginal API cost, own infrastructure.
    """
    return OpenAIEmbeddings(base_url=base_url, api_key=api_key, model=model)


def get_vectorstore(
    persist_dir: str, embedding_base_url: str, embedding_api_key: str, embedding_model: str
) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(embedding_base_url, embedding_api_key, embedding_model),
        persist_directory=persist_dir,
        # Without this, Chroma defaults to Euclidean distance, whose
        # relevance-score formula (1 - dist/sqrt(2)) assumes cosine
        # similarity is never negative — it isn't here, which produced
        # scores as low as -0.96 and made RETRIEVAL_SCORE_THRESHOLD
        # meaningless. Cosine distance makes the reported "relevance score"
        # equal to raw cosine similarity, matching how the threshold was
        # actually calibrated.
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
