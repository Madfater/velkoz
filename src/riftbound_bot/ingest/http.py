"""HTTP conventions shared by the card ingest sources.

Both card sources are read-only fetches against third-party endpoints that
neither of them controls, so they want the same politeness headers and the
same retry policy — kept here so the two can't drift apart.
"""
from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

REQUEST_HEADERS = {
    "User-Agent": "riftbound-bot-ingest/0.1 (personal, non-commercial data collection)",
    "Accept": "application/json",
}

MAX_ATTEMPTS = 4
# Status codes worth trying again: transient server-side failures and explicit
# rate limiting. Everything else in the 4xx range is a statement about the
# request itself and will fail identically however many times it's repeated.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def is_retryable(error: BaseException) -> bool:
    """Retry transport failures, 5xx, and 429 — never a deterministic 4xx.

    Retrying every HTTPStatusError meant the two documented failure modes of
    the Riftcodex API (403 without a User-Agent, 422 for an oversized `size`)
    were each retried four times with exponential backoff before surfacing
    the same error the first attempt already had.
    """
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in _RETRYABLE_STATUS_CODES
    return False


def retrying_request(func):
    """Applies the shared retry policy to a single HTTP fetch."""
    return retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )(func)
