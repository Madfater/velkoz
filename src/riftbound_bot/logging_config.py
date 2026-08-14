"""Structured logging setup shared by the bot and the ingest CLIs.

JSON output in production (easy to grep/ship to a log aggregator), a
human-readable console renderer when a TTY is attached (local dev).

Level comes from LOG_LEVEL. The default of INFO hides the per-pool score
range the chain logs at debug, which is the measurement needed to calibrate
RETRIEVAL_SCORE_THRESHOLD against a particular embedding endpoint — so
`LOG_LEVEL=DEBUG` is the supported way to get at it without editing code.
"""
from __future__ import annotations

import logging
import os
import sys

import structlog


def _configured_level(level: int | None) -> int:
    if level is not None:
        return level
    name = os.environ.get("LOG_LEVEL")
    if not name:
        return logging.INFO
    levels = logging.getLevelNamesMapping()
    resolved = levels.get(name.strip().upper())
    if resolved is None:
        raise RuntimeError(
            f"LOG_LEVEL={name!r} is not a log level "
            f"(expected one of {', '.join(sorted(levels))})"
        )
    return resolved


def configure_logging(level: int | None = None) -> None:
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    console = sys.stderr.isatty()
    renderer = (
        structlog.dev.ConsoleRenderer() if console else structlog.processors.JSONRenderer()
    )
    # ConsoleRenderer formats exceptions itself; JSONRenderer does not, and
    # without this it emitted `"exc_info": true` and dropped the traceback
    # entirely — so every error the bot logged in production (its only
    # failure reporting) arrived with no diagnostic content. Local dev has a
    # TTY and took the console path, which is why this stayed invisible.
    format_processors = (
        [] if console else [structlog.processors.format_exc_info]
    )

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *format_processors,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(_configured_level(level))
