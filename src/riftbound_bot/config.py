from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    discord_bot_token: str
    discord_guild_id: int
    generation_base_url: str
    generation_api_key: str
    generation_model: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    retrieval_pool_per_type: int
    retrieval_k: int
    retrieval_score_threshold: float
    vector_store_dir: str

    @classmethod
    def load(cls) -> Settings:
        return cls(
            discord_bot_token=_require("DISCORD_BOT_TOKEN"),
            discord_guild_id=int(_require("DISCORD_GUILD_ID")),
            **_generation_settings_kwargs(),
            **_embedding_settings_kwargs(),
            retrieval_pool_per_type=int(os.environ.get("RETRIEVAL_POOL_PER_TYPE", "10")),
            retrieval_k=int(os.environ.get("RETRIEVAL_K", "6")),
            retrieval_score_threshold=float(
                os.environ.get("RETRIEVAL_SCORE_THRESHOLD", "0.45")
            ),
            vector_store_dir=os.environ.get("VECTOR_STORE_DIR", "data/turbovec"),
        )

    @classmethod
    def load_for_ingest(cls) -> IngestSettings:
        """Ingestion only needs the embedding + storage config, not Discord/generation.

        `database_url`/`rules_dir` are ingest-time only — the live bot never
        touches Postgres or the rules Markdown source, only the built vector
        store (see rag/vectorstore.py's load_vectorstore).
        """
        return IngestSettings(
            **_embedding_settings_kwargs(),
            vector_store_dir=os.environ.get("VECTOR_STORE_DIR", "data/turbovec"),
            rules_dir=os.environ.get("RULES_DIR", "data/rules"),
            database_url=_require("DATABASE_URL"),
        )


def _generation_settings_kwargs() -> dict:
    return {
        "generation_base_url": _require("GENERATION_API_BASE_URL"),
        # Empty string means "no real key configured" — see rag/llm.py's
        # build_chat_model for why that needs special handling for keyless
        # gateways, rather than a placeholder baked in here.
        "generation_api_key": os.environ.get("GENERATION_API_KEY", ""),
        "generation_model": os.environ.get("GENERATION_MODEL", "deepseek-v4-flash-free"),
    }


def _embedding_settings_kwargs() -> dict:
    return {
        "embedding_base_url": _require("EMBEDDING_API_BASE_URL"),
        "embedding_api_key": _require("EMBEDDING_API_KEY"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "embedding"),
    }


@dataclass(frozen=True)
class IngestSettings:
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    vector_store_dir: str
    rules_dir: str
    database_url: str
