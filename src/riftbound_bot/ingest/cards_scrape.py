"""Scrapes card data from riftbound.chroniclecore.com (符文戰場編年史).

The site is a Next.js app. Its /gallery page turns out to embed the *entire*
card dataset as a JSON prop inside the React Server Component payload
(`self.__next_f.push(...)` chunks in the HTML) rather than requiring one
request per card — so this fetches a single page instead of ~1,256.

No official API or bulk export exists for this site (per the design doc);
this relies on that embedded data structure and will break if the site's
internal schema changes. If it does, use `cards_from_api.py` instead.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx
import structlog

from riftbound_bot.ingest.http import REQUEST_HEADERS, retrying_request
from riftbound_bot.ingest.models import CardRecord

logger = structlog.get_logger("riftbound_bot.cards_scrape")

GALLERY_URL = "https://riftbound.chroniclecore.com/gallery"
CARD_URL_TEMPLATE = "https://riftbound.chroniclecore.com/cards/{card_id}"

# The captured group is a JSON string literal, so the lazy quantifier stops at
# the first `"])` — a payload containing that sequence inside an escaped string
# would truncate the match and hand json.loads a partial literal. Not observed
# on this site, but it's the failure mode to suspect if decoding starts failing
# on a page that clearly still contains card data.
_PUSH_RE = re.compile(r"self\.__next_f\.push\(\[1,(\".*?\")\]\)", re.DOTALL)
_SEGMENT_RE = re.compile(r"\n(?=[0-9a-f]+:)")
_KEYWORD_MARKUP_RE = re.compile(r"\{\{(.*?)\}\}")


def _clean_effect_text(raw: str) -> str:
    """Strips the site's `{{keyword}}` highlight markup down to plain text."""
    return _KEYWORD_MARKUP_RE.sub(r"\1", raw).strip()


@retrying_request
def _fetch_gallery_html(client: httpx.Client) -> str:
    resp = client.get(GALLERY_URL, headers=REQUEST_HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _extract_rsc_payload(html: str) -> str:
    """Concatenates every `self.__next_f.push([1, "..."])` string chunk in
    document order — Next.js streams RSC data across multiple push calls, and
    a single logical segment can be split across chunk boundaries.
    """
    chunks = _PUSH_RE.findall(html)
    try:
        return "".join(json.loads(chunk) for chunk in chunks)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Could not decode the gallery page's RSC push chunks — the site's "
            "internal structure may have changed. Fall back to cards_from_api.py."
        ) from error


def _find_card_array(obj: Any) -> list[dict] | None:
    """Recursively searches the decoded RSC tree for the card list, identified
    by its distinctive shape rather than a hardcoded path (the exact nesting
    is an implementation detail of the page's component tree and could shift
    between deploys).
    """
    if isinstance(obj, list):
        if (
            obj
            and isinstance(obj[0], dict)
            and "zh_hant" in obj[0]
            and "stats" in obj[0]
            and len(obj) > 50
        ):
            return obj
        for item in obj:
            found = _find_card_array(item)
            if found is not None:
                return found
    elif isinstance(obj, dict):
        for value in obj.values():
            found = _find_card_array(value)
            if found is not None:
                return found
    return None


def _extract_card_dicts(html: str) -> list[dict]:
    payload = _extract_rsc_payload(html)
    segments = _SEGMENT_RE.split(payload)
    for segment in segments:
        _, _, value = segment.partition(":")
        value = value.strip()
        if not value or not value.startswith(("[", "{")):
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            # Most segments legitimately aren't JSON; log enough to tell that
            # apart from the card segment itself failing to decode, which
            # would otherwise surface as the misleading "structure changed".
            logger.debug(
                "cards_scrape.segment_skipped", prefix=value[:60], error=str(error)
            )
            continue
        found = _find_card_array(decoded)
        if found is not None:
            return found
    raise RuntimeError(
        "Could not locate the card data array in the gallery page's RSC "
        "payload — the site's internal structure may have changed. "
        "Fall back to cards_from_api.py."
    )


def normalize_card(raw: dict) -> CardRecord:
    card_id: str = raw["id"]
    set_code, _, collector_number = card_id.partition("-")
    stats = raw.get("stats") or {}
    zh_hant = raw.get("zh_hant") or {}
    en = raw.get("en") or {}
    return CardRecord(
        id=card_id,
        set=set_code,
        collector_number=collector_number,
        name_zh=zh_hant.get("name", ""),
        name_en=en.get("name", ""),
        category=stats.get("category", ""),
        color=stats.get("color", ""),
        energy=stats.get("energy"),
        power=stats.get("power"),
        might=stats.get("might"),
        rarity=stats.get("rarity", ""),
        tags=raw.get("tags") or [],
        rules_text_zh=_clean_effect_text(zh_hant.get("effect", "")),
        source_url=CARD_URL_TEMPLATE.format(card_id=card_id),
    )


def scrape_all_cards() -> list[CardRecord]:
    with httpx.Client() as client:
        html = _fetch_gallery_html(client)
    raw_cards = _extract_card_dicts(html)
    return [normalize_card(raw) for raw in raw_cards]


def main() -> None:
    from riftbound_bot.config import load_ingest_settings
    from riftbound_bot.ingest.db import get_connection, upsert_card_records
    from riftbound_bot.logging_config import configure_logging

    configure_logging()
    settings = load_ingest_settings()
    cards = scrape_all_cards()
    with get_connection(settings) as conn:
        upserted = upsert_card_records(conn, cards)
    logger.info("cards_scrape.done", upserted=upserted)


if __name__ == "__main__":
    main()
