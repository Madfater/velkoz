"""Fallback card data source: the Riftcodex community REST API
(https://riftcodex.com), English, translated via the configured generation
model.

Use this only if `cards_scrape.py` breaks (chroniclecore.com changes its
internal data shape) — the design doc treats chroniclecore.com as the
primary source since it's already native Traditional Chinese and needs no
translation. This path costs generation-model calls, one batch request per
`BATCH_SIZE` cards.

This replaces an earlier fallback that read OwenMelbz's card-data gist. That
gist was a single frozen snapshot (one revision, committed 2025-11-15, never
updated): 394 cards covering only OGN/OGS and an 18-card slice of SFD, so it
had fallen three full sets behind. Riftcodex is a maintained REST API with no
auth, covering all 8 sets — it listed Vendetta within days of that set's
2026-07-31 release.

Riot's own gallery (playriftbound.com) was the other candidate. It publishes
the same data in a single `__NEXT_DATA__` blob, but ships only de/en/es/fr/
it/ja/ko — no Chinese at all — and scraping it would leave the project with a
second brittle Next.js scrape as the backup for its first one. The API is the
better hedge precisely because it fails differently.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from riftbound_bot.ingest.cards_scrape import CardRecord
from riftbound_bot.rag.llm import build_chat_model

API_CARDS_URL = "https://api.riftcodex.com/cards"
CARD_URL_TEMPLATE = "https://riftcodex.com/cards/{card_id}"
# The API rejects `size` above 100 with a 422, and 403s a request that sends
# no User-Agent at all.
PAGE_SIZE = 100
BATCH_SIZE = 20

REQUEST_HEADERS = {
    "User-Agent": "riftbound-bot-ingest/0.1 (personal, non-commercial data collection)",
    "Accept": "application/json",
}

# chroniclecore stores `color` as an English colour word rather than the
# domain name, so the fallback matches that vocabulary to keep both sources
# writing the same values into the same column.
_DOMAIN_COLORS = {
    "Fury": "red",
    "Calm": "green",
    "Mind": "blue",
    "Body": "orange",
    "Chaos": "purple",
    "Order": "yellow",
    "Colorless": "colorless",
}

# Category and rarity are Traditional Chinese on the primary source; these are
# the exact strings chroniclecore emits. Riftcodex's `type` is coarser (it has
# no separate champion-unit/signature/token categories), so this maps onto the
# subset it can express.
_TYPE_CATEGORIES = {
    "Unit": "單位",
    "Spell": "法術",
    "Gear": "裝備",
    "Legend": "傳奇",
    "Battlefield": "戰場",
    "Rune": "符文",
}

_RARITIES = {
    "Common": "普通",
    "Uncommon": "不凡",
    "Rare": "稀有",
    "Epic": "史詩",
    "Showcase": "異畫",
    "Promo": "宣傳卡",
}

# Trait tags, verified against the 41 distinct tags chroniclecore emits.
# Riftcodex also tags cards with champion names (Zed, Master Yi, ...); those
# are passed through untranslated rather than guessed at — see `_map_tags`.
_TRAIT_TAGS = {
    "Ionia": "艾歐尼亞",
    "Noxus": "諾克薩斯",
    "Shurima": "恕瑞瑪",
    "Bilgewater": "比爾吉沃特",
    "Demacia": "蒂瑪西亞",
    "Bandle City": "班德爾城",
    "Mount Targon": "巨石峰",
    "Freljord": "弗雷爾卓德",
    "Piltover": "皮爾特沃夫",
    "Zaun": "佐恩",
    "Shadow Isles": "暗影島",
    "The Void": "虛空之地",
    "Ixtal": "以緒塔爾",
    "Icathia": "艾卡西亞",
    "Kathkan": "喀斯坎",
    "Yordle": "約德爾人",
    "Equipment": "武裝",
    "Dog": "犬形",
    "Dragon": "龍",
    "Pirate": "海盜",
    "Poro": "普羅",
    "Fae": "妖精",
    "Cat": "貓科",
    "Bird": "鳥類",
    "Mech": "機械",
    "Robot": "機器人",
    "Spider": "蜘蛛",
    "Spirit": "靈體",
    "Demon": "惡魔",
    "Elite": "精銳",
    "Sentinel": "哨兵",
    "Recruit": "隨從",
    "Trifarian": "特菲利安",
}

# The API's text carries the game's icon placeholders verbatim. Expand them to
# words so the translation model sees prose rather than `:rb_*:` tokens, and so
# the embedded text stays searchable.
_ICON_RE = re.compile(r":rb_([a-z0-9_]+):")
_ICON_WORDS = {
    "might": "Might",
    "exhaust": "Exhaust",
    "rune_rainbow": "any rune",
}

# `text.rich` uses these to separate one rules line from the next; `text.plain`
# drops them without substituting anything, which runs the end of one ability
# into the start of the next ("...spell or ability.)When you play me...").
_LINE_BREAK_RE = re.compile(r"<br\s*/?>|</p>|</li>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Printing qualifiers the API appends to a name. They describe the physical
# printing, not the card, so they are dropped rather than handed to the
# translation model. Matched only as a trailing parenthetical and only against
# this list, so real names keep theirs (`Sprite (274) // Buff`).
_PRINTING_QUALIFIERS = frozenset(
    {
        "Alternate Art",
        "Overnumbered",
        "Signature",
        "Metal",
        "Starter",
        "Ultimate",
        "Launch Exclusive",
        "GG EZ",
    }
)
_TRAILING_PAREN_RE = re.compile(r"\s*\(([^)]*)\)\s*$")

_TRANSLATION_SYSTEM_PROMPT = """\
你是 Riftbound 集換式卡牌遊戲的繁體中文在地化人員。將下列英文卡牌名稱與規則文字翻譯成
繁體中文，用語需與官方 TCG 規則書的風格一致（簡潔、精確、避免意譯）。

保留方括號關鍵字（例如 [Accelerate]、[Deflect]）不要翻譯方括號本身的格式，但關鍵字名稱本身
請翻成對應的繁體中文術語（Accelerate -> 疾行、Deflect -> 偏斜、Action -> 行動、
Reaction -> 反應、Equip -> 裝備、Quick-Draw -> 快拔）；遇到其他未列出的關鍵字，選擇最貼近
字面意義且簡短的翻譯。

規則文字中的數值與資源術語請照下列對應翻譯：Might -> 戰力、Energy -> 費用、rune -> 符文、
any rune -> 任意符文、Exhaust -> 橫置。

輸入是一個 JSON 陣列，每個元素是 {"id": "...", "name": "...", "text": "..."}。
輸出必須是「同樣長度、同樣順序」的 JSON 陣列，每個元素是
{"id": "...", "name_zh": "...", "text_zh": "..."}，不要添加任何其他文字或說明。
"""


@dataclass(frozen=True)
class ApiCard:
    id: str
    name_en: str
    text_en: str
    set: str
    collector_number: str
    category: str
    color: str
    rarity: str
    energy: int | None
    power: int | None
    might: int | None
    tags: list[str] = field(default_factory=list)
    image_url: str = ""


def _expand_icons(text: str) -> str:
    """Turns `:rb_might:` / `:rb_energy_2:` / `:rb_rune_fury:` into words."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token in _ICON_WORDS:
            return f" {_ICON_WORDS[token]} "
        if token.startswith("energy_"):
            return f" {token.removeprefix('energy_')} Energy "
        if token.startswith("rune_"):
            return f" {token.removeprefix('rune_').capitalize()} rune "
        return " "

    return _ICON_RE.sub(replace, text)


def _clean_text(raw: str) -> str:
    """Renders `text.rich` down to plain text, one rules line per line.

    Built from `rich` rather than `plain` because only `rich` marks where one
    rules line ends and the next begins.
    """
    text = _LINE_BREAK_RE.sub("\n", raw or "")
    text = _HTML_TAG_RE.sub("", text)
    text = _expand_icons(html.unescape(text))
    lines = []
    for line in text.split("\n"):
        # Icon expansion pads its replacements, which can leave a space sitting
        # in front of punctuation ("Fury rune , Exhaust :").
        line = re.sub(r"[ \t]+", " ", line)
        line = re.sub(r" ([,.:;!?)])", r"\1", line)
        line = re.sub(r"([(]) ", r"\1", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _clean_name(raw: str) -> str:
    match = _TRAILING_PAREN_RE.search(raw or "")
    if match and match.group(1) in _PRINTING_QUALIFIERS:
        return _TRAILING_PAREN_RE.sub("", raw).strip()
    return (raw or "").strip()


def _printing_rank(raw: dict[str, Any]) -> tuple[bool, bool, bool, str]:
    """Sort key that prefers a card's base printing over its alternate-art,
    signature, and overnumbered reprints.

    This is not just cosmetic: alternate-art printings omit the parenthetical
    reminder text that spells out each keyword, so the base printing carries
    strictly more of what a rules bot needs to retrieve.
    """
    meta = raw.get("metadata") or {}
    return (
        bool(meta.get("alternate_art")),
        bool(meta.get("signature")),
        bool(meta.get("overnumbered")),
        str(raw.get("tcgplayer_id") or ""),
    )


def _map_tags(tags: list[str]) -> list[str]:
    return [_TRAIT_TAGS.get(tag, tag) for tag in tags]


def _normalize_card(raw: dict[str, Any]) -> ApiCard:
    attributes = raw.get("attributes") or {}
    classification = raw.get("classification") or {}
    text = raw.get("text") or {}
    set_info = raw.get("set") or {}
    media = raw.get("media") or {}

    set_code = str(set_info.get("set_id") or "")
    collector_number = str(raw.get("collector_number") or "").zfill(3)
    domains = classification.get("domain") or []
    return ApiCard(
        id=f"{set_code}-{collector_number}",
        name_en=_clean_name(raw.get("name") or ""),
        text_en=_clean_text(text.get("rich") or ""),
        set=set_code,
        collector_number=collector_number,
        category=_TYPE_CATEGORIES.get(classification.get("type") or "", ""),
        # Dual-domain cards (mostly Legends) keep both, since dropping one
        # would misstate what the card can be played with.
        color="/".join(_DOMAIN_COLORS.get(d, d.lower()) for d in domains),
        rarity=_RARITIES.get(classification.get("rarity") or "", ""),
        energy=attributes.get("energy"),
        power=attributes.get("power"),
        might=attributes.get("might"),
        tags=_map_tags(raw.get("tags") or []),
        # Already absolute (Riot's own CDN), unlike the primary scraper's
        # site-relative asset paths. This face is English — this source has no
        # Chinese art at all, which is the same reason it needs an LLM to
        # produce Chinese rules text.
        image_url=str(media.get("image_url") or ""),
    )


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _fetch_page(client: httpx.Client, page: int) -> dict[str, Any]:
    resp = client.get(
        API_CARDS_URL,
        params={"size": PAGE_SIZE, "page": page},
        headers=REQUEST_HEADERS,
        timeout=30,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()


def _dedupe_printings(raw_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapses the API's per-printing rows to one row per printed card.

    The API returns a row per physical printing — foil/Metal reprints, alternate
    art, signature Legends — so several rows can share one `SET-NNN` number.
    Collapsing them matches the primary source's one-row-per-card granularity
    and keeps near-duplicate rules text out of the vector index.
    """
    best: dict[str, dict[str, Any]] = {}
    for raw in raw_cards:
        card_id = (
            f"{(raw.get('set') or {}).get('set_id') or ''}-"
            f"{str(raw.get('collector_number') or '').zfill(3)}"
        )
        if card_id not in best or _printing_rank(raw) < _printing_rank(best[card_id]):
            best[card_id] = raw
    return list(best.values())


def fetch_api_cards() -> list[ApiCard]:
    """Pages through the whole card list, keeping one row per printed card."""
    raw_cards: list[dict[str, Any]] = []
    with httpx.Client() as client:
        page = 1
        while True:
            payload = _fetch_page(client, page)
            raw_cards.extend(payload.get("items") or [])
            if page >= (payload.get("pages") or 1):
                break
            page += 1

    if not raw_cards:
        raise RuntimeError(
            f"{API_CARDS_URL} returned no cards — the API may have changed "
            "shape or gone away."
        )

    return [_normalize_card(raw) for raw in _dedupe_printings(raw_cards)]


def _translate_batch(llm: BaseChatModel, batch: list[ApiCard]) -> dict[str, tuple[str, str]]:
    payload = [{"id": c.id, "name": c.name_en, "text": c.text_en} for c in batch]
    response = llm.invoke(
        [
            ("system", _TRANSLATION_SYSTEM_PROMPT),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ]
    )
    translated = json.loads(str(response.content))
    return {item["id"]: (item["name_zh"], item["text_zh"]) for item in translated}


def translate_all(
    cards: list[ApiCard], base_url: str, api_key: str, model: str
) -> list[CardRecord]:
    llm = build_chat_model(base_url=base_url, api_key=api_key, model=model)
    records: list[CardRecord] = []
    for start in range(0, len(cards), BATCH_SIZE):
        batch = cards[start : start + BATCH_SIZE]
        translations = _translate_batch(llm, batch)
        for card in batch:
            name_zh, text_zh = translations.get(card.id, (card.name_en, card.text_en))
            records.append(
                CardRecord(
                    id=card.id,
                    set=card.set,
                    collector_number=card.collector_number,
                    name_zh=name_zh,
                    name_en=card.name_en,
                    category=card.category,
                    color=card.color,
                    energy=card.energy,
                    power=card.power,
                    might=card.might,
                    rarity=card.rarity,
                    tags=card.tags,
                    rules_text_zh=text_zh,
                    source_url=CARD_URL_TEMPLATE.format(card_id=card.id),
                    image_url=card.image_url,
                )
            )
    return records


def main() -> None:
    from riftbound_bot.config import Settings
    from riftbound_bot.ingest.db import get_connection, upsert_cards
    from riftbound_bot.logging_config import configure_logging

    configure_logging()
    # Generation config lives on the bot's full Settings (Discord token
    # required to load it, even though this script doesn't touch Discord —
    # a pre-existing quirk, not something this migration changes);
    # database_url is ingest-only, so it's loaded separately from IngestSettings.
    bot_settings = Settings.load()
    ingest_settings = Settings.load_for_ingest()
    cards = fetch_api_cards()
    records = translate_all(
        cards,
        base_url=bot_settings.generation_base_url,
        api_key=bot_settings.generation_api_key,
        model=bot_settings.generation_model,
    )
    with get_connection(ingest_settings) as conn:
        upsert_cards(conn, [r.__dict__ for r in records])
    print(f"Upserted {len(records)} translated cards into Postgres.")


if __name__ == "__main__":
    main()
