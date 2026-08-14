"""Environment-backed settings for the bot and the ingest CLIs.

Split by what each entrypoint actually needs rather than one settings object
for everything: the bot needs Discord plus generation plus embeddings, the
ingest CLIs need embeddings plus storage, and card translation needs only
the generation config. Loading more than that would make unrelated
credentials mandatory — the card-translation script used to require a
Discord bot token.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

DEFAULT_VECTOR_STORE_DIR = "data/turbovec"


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        # Naming the variable matters: a bare int() ValueError says only that
        # some string wasn't a number, with no clue which setting to fix.
        raise RuntimeError(f"{name} must be a whole number, got {raw!r}") from None


def _require_int(name: str) -> int:
    raw = _require(name)
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a whole number, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    # repr=False on the credentials: this dataclass would otherwise print the
    # bot token in any log line, traceback frame, or debugger view holding it.
    discord_bot_token: str = field(repr=False)
    discord_guild_id: int
    generation_base_url: str
    generation_api_key: str = field(repr=False)
    generation_model: str
    embedding_base_url: str
    embedding_api_key: str = field(repr=False)
    embedding_model: str
    retrieval_pool_per_type: int
    retrieval_k: int
    retrieval_score_threshold: float
    vector_store_dir: str

    @classmethod
    def load(cls) -> Settings:
        return cls(
            discord_bot_token=_require("DISCORD_BOT_TOKEN"),
            discord_guild_id=_require_int("DISCORD_GUILD_ID"),
            **_generation_settings_kwargs(),
            **_embedding_settings_kwargs(),
            retrieval_pool_per_type=_env_int("RETRIEVAL_POOL_PER_TYPE", 10),
            retrieval_k=_env_int("RETRIEVAL_K", 6),
            retrieval_score_threshold=_env_float("RETRIEVAL_SCORE_THRESHOLD", 0.5),
            vector_store_dir=_vector_store_dir(),
        )

    @classmethod
    def load_generation(cls) -> GenerationSettings:
        """Generation config on its own, with no Discord credentials required.

        cards_from_api.py translates card text with the generation model but
        never touches Discord — loading the full Settings just to reach
        generation_* would make a bot token mandatory to run a card ingest.
        """
        return GenerationSettings(**_generation_settings_kwargs())


@dataclass(frozen=True)
class GenerationSettings:
    generation_base_url: str
    generation_api_key: str = field(repr=False)
    generation_model: str


@dataclass(frozen=True)
class IngestSettings:
    embedding_base_url: str
    embedding_api_key: str = field(repr=False)
    embedding_model: str
    vector_store_dir: str
    rules_dir: str
    database_url: str = field(repr=False)


def load_ingest_settings() -> IngestSettings:
    """Ingestion only needs the embedding + storage config, not Discord/generation.

    `database_url`/`rules_dir` are ingest-time only — the live bot never
    touches Postgres or the rules Markdown source, only the built vector
    store (see rag/vectorstore.py's load_vectorstore).
    """
    return IngestSettings(
        **_embedding_settings_kwargs(),
        vector_store_dir=_vector_store_dir(),
        rules_dir=os.environ.get("RULES_DIR", "data/rules"),
        database_url=_require("DATABASE_URL"),
    )


def _vector_store_dir() -> str:
    return os.environ.get("VECTOR_STORE_DIR", DEFAULT_VECTOR_STORE_DIR)


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
