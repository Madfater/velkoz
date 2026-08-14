"""Covers settings loading, which every entrypoint depends on.

config.load_dotenv() runs at import time, so these tests clear the variables
they care about rather than assuming a clean environment.
"""
import pytest

from riftbound_bot.config import Settings, load_ingest_settings

_BOT_ENV = {
    "DISCORD_BOT_TOKEN": "token",
    "DISCORD_GUILD_ID": "123",
    "GENERATION_API_BASE_URL": "http://generation.test",
    "EMBEDDING_API_BASE_URL": "http://embedding.test",
    "EMBEDDING_API_KEY": "key",
}


@pytest.fixture
def bot_env(monkeypatch):
    for name in (
        "GENERATION_API_KEY",
        "GENERATION_MODEL",
        "EMBEDDING_MODEL",
        "RETRIEVAL_POOL_PER_TYPE",
        "RETRIEVAL_K",
        "RETRIEVAL_SCORE_THRESHOLD",
        "VECTOR_STORE_DIR",
        "RULES_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in _BOT_ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_settings_load_applies_documented_defaults(bot_env):
    settings = Settings.load()

    assert settings.discord_guild_id == 123
    assert settings.retrieval_pool_per_type == 10
    assert settings.retrieval_k == 6
    # 0.5 is TurboVec's cosine-0 point; see the chain's threshold handling.
    assert settings.retrieval_score_threshold == 0.5
    assert settings.vector_store_dir == "data/turbovec"


def test_a_missing_required_variable_names_itself(bot_env):
    bot_env.delenv("DISCORD_BOT_TOKEN")

    with pytest.raises(RuntimeError, match="DISCORD_BOT_TOKEN"):
        Settings.load()


def test_a_non_numeric_setting_names_itself(bot_env):
    # A bare int() ValueError said only that some string wasn't a number.
    bot_env.setenv("RETRIEVAL_K", "六")

    with pytest.raises(RuntimeError, match="RETRIEVAL_K"):
        Settings.load()


def test_a_non_numeric_guild_id_names_itself(bot_env):
    bot_env.setenv("DISCORD_GUILD_ID", "not-a-snowflake")

    with pytest.raises(RuntimeError, match="DISCORD_GUILD_ID"):
        Settings.load()


def test_secrets_are_kept_out_of_the_repr(bot_env):
    bot_env.setenv("GENERATION_API_KEY", "super-secret-generation-key")
    settings = Settings.load()

    rendered = repr(settings)
    assert "token" not in rendered
    assert "super-secret-generation-key" not in rendered
    assert "123" in rendered  # non-secret fields still shown


def test_generation_settings_load_without_discord_credentials(bot_env):
    # Translating cards is not a Discord operation; requiring a bot token to
    # run the ingest script was the previous behaviour.
    bot_env.delenv("DISCORD_BOT_TOKEN")
    bot_env.delenv("DISCORD_GUILD_ID")

    settings = Settings.load_generation()

    assert settings.generation_base_url == "http://generation.test"
    assert settings.generation_model == "deepseek-v4-flash-free"


def test_ingest_settings_need_no_discord_or_generation_config(bot_env):
    bot_env.delenv("DISCORD_BOT_TOKEN")
    bot_env.delenv("GENERATION_API_BASE_URL")
    bot_env.setenv("DATABASE_URL", "postgresql://localhost/x")

    settings = load_ingest_settings()

    assert settings.rules_dir == "data/rules"
    assert settings.vector_store_dir == "data/turbovec"
