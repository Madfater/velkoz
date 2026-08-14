"""Covers the production (non-TTY) logging path.

pytest captures stderr, so `sys.stderr.isatty()` is False here — these tests
exercise the JSON renderer the bot actually runs under in Docker, which is
the path where exception detail used to disappear.
"""
import json
import logging

import pytest
import structlog

from riftbound_bot.logging_config import configure_logging


@pytest.fixture(autouse=True)
def _restore_logging():
    yield
    structlog.reset_defaults()
    logging.getLogger().handlers = []


def _emit_exception(capsys) -> dict:
    configure_logging()
    logger = structlog.get_logger("test")
    try:
        raise ValueError("card lookup exploded")
    except ValueError:
        logger.exception("bot.thing_failed", channel_id=7)
    return json.loads(capsys.readouterr().err.strip().splitlines()[-1])


def test_json_logs_carry_the_traceback(capsys):
    # This used to render as `"exc_info": true` with no type, message, or
    # frames — the bot's only error reporting, with the detail stripped out.
    record = _emit_exception(capsys)

    assert record["event"] == "bot.thing_failed"
    assert record["channel_id"] == 7
    assert "ValueError: card lookup exploded" in record["exception"]
    assert "Traceback" in record["exception"]


def test_log_level_comes_from_the_environment(monkeypatch, capsys):
    # The chain logs its per-pool score range at debug; that measurement is
    # how RETRIEVAL_SCORE_THRESHOLD gets calibrated, so it has to be
    # reachable without a code edit.
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging()
    structlog.get_logger("test").debug("chain.search_pool", score_max=0.61)

    assert "chain.search_pool" in capsys.readouterr().err


def test_debug_logs_are_suppressed_by_default(monkeypatch, capsys):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    configure_logging()
    structlog.get_logger("test").debug("chain.search_pool")

    assert capsys.readouterr().err.strip() == ""


def test_an_unusable_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "chatty")

    with pytest.raises(RuntimeError, match="LOG_LEVEL"):
        configure_logging()
