"""Scrapes card data from riftbound.chroniclecore.com (符文戰場編年史).

The site is a Next.js app. Its /gallery page turns out to embed the *entire*
card dataset as a JSON prop inside the React Server Component payload
(`self.__next_f.push(...)` chunks in the HTML) rather than requiring one
request per card — so this fetches a single page instead of ~1,256.

No official API or bulk export exists for this site (per the design doc);
this relies on that embedded data structure and will break if the site's
internal schema changes. If it does, use `cards_from_gist.py` instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

GALLERY_URL = "https://riftbound.chroniclecore.com/gallery"
CARD_URL_TEMPLATE = "https://riftbound.chroniclecore.com/cards/{card_id}"

_PUSH_RE = re.compile(r"self\.__next_f\.push\(\[1,(\".*?\")\]\)", re.DOTALL)
_SEGMENT_RE = re.compile(r"\n(?=[0-9a-f]+:)")
_KEYWORD_MARKUP_RE = re.compile(r"\{\{(.*?)\}\}")

REQUEST_HEADERS = {
    "User-Agent": "riftbound-bot-ingest/0.1 (personal, non-commercial data collection)"
}


@dataclass(frozen=True)
class CardRecord:
    id: str
    set: str
    collector_number: str
    name_zh: str
    name_en: str
    category: str
    color: str
    energy: int | None
    power: int | None
    might: int | None
    rarity: str
    tags: list[str]
    rules_text_zh: str
    source_url: str

    @property
    def text(self) -> str:
        """Text to embed for retrieval."""
        stat_bits = []
        if self.energy is not None:
            stat_bits.append(f"費用 {self.energy}")
        if self.power is not None:
            stat_bits.append(f"力量 {self.power}")
        if self.might is not None:
            stat_bits.append(f"戰力 {self.might}")
        stats = "、".join(stat_bits)
        tag_str = f"｜標籤：{'、'.join(self.tags)}" if self.tags else ""
        return (
            f"{self.name_zh}（{self.name_en}）｜{self.id}｜{self.category}｜"
            f"顏色：{self.color}｜稀有度：{self.rarity}｜{stats}{tag_str}\n"
            f"效果：{self.rules_text_zh}"
        )


def _clean_effect_text(raw: str) -> str:
    """Strips the site's `{{keyword}}` highlight markup down to plain text."""
    return _KEYWORD_MARKUP_RE.sub(r"\1", raw).strip()


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
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
    return "".join(json.loads(chunk) for chunk in chunks)


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
        except json.JSONDecodeError:
            continue
        found = _find_card_array(decoded)
        if found is not None:
            return found
    raise RuntimeError(
        "Could not locate the card data array in the gallery page's RSC "
        "payload — the site's internal structure may have changed. "
        "Fall back to cards_from_gist.py."
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
    from riftbound_bot.config import Settings
    from riftbound_bot.ingest.db import get_connection, upsert_cards
    from riftbound_bot.logging_config import configure_logging

    configure_logging()
    settings = Settings.load_for_ingest()
    cards = scrape_all_cards()
    records = [card.__dict__ for card in cards]
    with get_connection(settings) as conn:
        upsert_cards(conn, records)
    print(f"Upserted {len(records)} cards into Postgres.")


if __name__ == "__main__":
    main()
