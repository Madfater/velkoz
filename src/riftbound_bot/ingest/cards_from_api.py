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
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from langchain_core.language_models.chat_models import BaseChatModel

from riftbound_bot.ingest.cards_scrape import CardRecord
from riftbound_bot.ingest.http import REQUEST_HEADERS, retrying_request
from riftbound_bot.rag.llm import build_chat_model

logger = structlog.get_logger("riftbound_bot.cards_from_api")

API_CARDS_URL = "https://api.riftcodex.com/cards"
CARD_URL_TEMPLATE = "https://riftcodex.com/cards/{card_id}"
# The API rejects `size` above 100 with a 422, and 403s a request that sends
# no User-Agent at all.
PAGE_SIZE = 100
BATCH_SIZE = 20
# How many times one batch's translation is re-requested before giving up.
TRANSLATION_ATTEMPTS = 3
# Backstop against a malformed `pages` value turning pagination into an
# unbounded loop. The corpus is ~1,131 cards over ~12 pages today.
MAX_PAGES = 200

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


def _printing_rank(raw: dict[str, Any]) -> tuple[bool, bool, bool, bool, bool, int]:
    """Sort key that prefers a card's base printing over its reprints.

    This is not just cosmetic: alternate-art printings omit the parenthetical
    reminder text that spells out each keyword, so the base printing carries
    strictly more of what a rules bot needs to retrieve.

    `metadata` only flags three of the eight printing variants this API emits;
    the rest (Metal, Starter, Ultimate, Launch Exclusive, GG EZ) announce
    themselves only in the trailing parenthetical of the name, which is what
    the qualifier check picks up. Without it a Metal reprint carried none of
    the ranked flags and won or lost on the tiebreak alone.
    """
    meta = raw.get("metadata") or {}
    name = raw.get("name") or ""
    tcgplayer_id = raw.get("tcgplayer_id")
    try:
        # Numeric: as strings "10" sorts before "9". Missing ids sort last so a
        # printing with an id always beats one without.
        rank_id = int(tcgplayer_id)
    except (TypeError, ValueError):
        rank_id = None
    return (
        bool(meta.get("alternate_art")),
        bool(meta.get("signature")),
        bool(meta.get("overnumbered")),
        _clean_name(name) != name.strip(),
        rank_id is None,
        rank_id if rank_id is not None else 0,
    )


def _map_tags(tags: list[str]) -> list[str]:
    return [_TRAIT_TAGS.get(tag, tag) for tag in tags]


def _collector_number(raw: dict[str, Any]) -> str:
    # Explicitly against None: `or ""` turned a collector number of 0 into
    # "000" via the empty string rather than through zfill.
    number = raw.get("collector_number")
    return ("" if number is None else str(number)).zfill(3)


def _card_id(raw: dict[str, Any]) -> str:
    """The `SET-NNN` id both the dedupe pass and normalization key on.

    Shared so the two can't derive it differently — they did, and a divergence
    would silently make dedupe group by something other than the final id.
    """
    set_code = str((raw.get("set") or {}).get("set_id") or "")
    return f"{set_code}-{_collector_number(raw)}"


def _normalize_card(raw: dict[str, Any]) -> ApiCard:
    attributes = raw.get("attributes") or {}
    classification = raw.get("classification") or {}
    text = raw.get("text") or {}
    set_info = raw.get("set") or {}

    set_code = str(set_info.get("set_id") or "")
    collector_number = _collector_number(raw)
    domains = classification.get("domain") or []
    return ApiCard(
        id=_card_id(raw),
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
    )


@retrying_request
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
        card_id = _card_id(raw)
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
            items = payload.get("items") or []
            raw_cards.extend(items)
            total_pages = payload.get("pages")
            if total_pages is None:
                # Without this the old `or 1` treated a renamed/absent key as
                # "one page total" and returned the first 100 cards as if that
                # were the whole corpus.
                logger.warning("cards_from_api.missing_page_count", page=page, fetched=len(raw_cards))
            logger.info("cards_from_api.fetched_page", page=page, of=total_pages, cards=len(items))
            reached_last_page = total_pages is not None and page >= total_pages
            if not items or reached_last_page or page >= MAX_PAGES:
                break
            page += 1

    if not raw_cards:
        raise RuntimeError(
            f"{API_CARDS_URL} returned no cards — the API may have changed "
            "shape or gone away."
        )

    return [_normalize_card(raw) for raw in _dedupe_printings(raw_cards)]


class TranslationBatchError(RuntimeError):
    """A translation response that can't be trusted to describe this batch."""


def _strip_code_fences(content: str) -> str:
    """Models routinely wrap JSON in ```json fences despite being told not to."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    without_open = re.sub(r"^```[A-Za-z]*\s*", "", text)
    return re.sub(r"\s*```$", "", without_open).strip()


def _parse_batch_response(content: str, batch: list[ApiCard]) -> dict[str, tuple[str, str]]:
    """Parses one batch's response, rejecting anything that doesn't describe
    exactly the cards that were sent.

    The ids matter more than they look: translations used to be applied with
    `translations.get(card.id, (card.name_en, card.text_en))`, so a model that
    echoed the wrong ids (or returned a short list) silently wrote *English*
    into name_zh/rules_text_zh for every unmatched card. That produces a
    Chinese-language index full of English text with nothing in the logs, so
    an unusable batch is now an error rather than a quiet downgrade.
    """
    try:
        translated = json.loads(_strip_code_fences(content))
    except json.JSONDecodeError as error:
        raise TranslationBatchError(f"response was not valid JSON: {error}") from error

    if not isinstance(translated, list):
        raise TranslationBatchError(f"expected a JSON array, got {type(translated).__name__}")

    parsed: dict[str, tuple[str, str]] = {}
    for item in translated:
        if not isinstance(item, dict):
            raise TranslationBatchError(f"expected objects in the array, got {type(item).__name__}")
        missing = {"id", "name_zh", "text_zh"} - item.keys()
        if missing:
            raise TranslationBatchError(f"item is missing {sorted(missing)}")
        parsed[str(item["id"])] = (str(item["name_zh"]), str(item["text_zh"]))

    expected = {card.id for card in batch}
    if parsed.keys() != expected:
        raise TranslationBatchError(
            f"ids don't match the batch: missing {sorted(expected - parsed.keys())}, "
            f"unexpected {sorted(parsed.keys() - expected)}"
        )
    return parsed


def _translate_batch(llm: BaseChatModel, batch: list[ApiCard]) -> dict[str, tuple[str, str]]:
    """Translates one batch, retrying a response that fails validation.

    Retries here rather than around the whole run: a malformed response is
    usually a one-off sampling artifact, and re-asking for the same batch is
    far cheaper than restarting a ~57-batch translation.
    """
    payload = [{"id": c.id, "name": c.name_en, "text": c.text_en} for c in batch]
    messages = [
        ("system", _TRANSLATION_SYSTEM_PROMPT),
        ("human", json.dumps(payload, ensure_ascii=False)),
    ]
    for attempt in range(1, TRANSLATION_ATTEMPTS + 1):
        response = llm.invoke(messages)
        try:
            return _parse_batch_response(str(response.content), batch)
        except TranslationBatchError as error:
            if attempt == TRANSLATION_ATTEMPTS:
                raise
            logger.warning(
                "cards_from_api.translation_batch_invalid",
                attempt=attempt,
                of=TRANSLATION_ATTEMPTS,
                error=str(error),
            )
    raise AssertionError("unreachable")


def _to_record(card: ApiCard, name_zh: str, text_zh: str) -> CardRecord:
    return CardRecord(
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
    )


def translate_batches(
    cards: list[ApiCard], base_url: str, api_key: str, model: str
) -> Iterator[list[CardRecord]]:
    """Yields one translated batch at a time so callers can persist as they go.

    Translating everything before the first write meant a failure in the last
    batch discarded every generation call made before it; upserts are
    idempotent, so persisting per batch makes a re-run resume cheaply instead.
    """
    llm = build_chat_model(base_url=base_url, api_key=api_key, model=model)
    total_batches = (len(cards) + BATCH_SIZE - 1) // BATCH_SIZE
    for index, start in enumerate(range(0, len(cards), BATCH_SIZE), start=1):
        batch = cards[start : start + BATCH_SIZE]
        translations = _translate_batch(llm, batch)
        logger.info("cards_from_api.translated_batch", batch=index, of=total_batches, cards=len(batch))
        yield [_to_record(card, *translations[card.id]) for card in batch]


def main() -> None:
    from riftbound_bot.config import Settings
    from riftbound_bot.ingest.db import get_connection, upsert_cards
    from riftbound_bot.logging_config import configure_logging

    configure_logging()
    generation_settings = Settings.load_generation()
    ingest_settings = Settings.load_for_ingest()
    cards = fetch_api_cards()

    # Upsert each batch as it lands rather than after the whole translation:
    # a failure part-way through then keeps everything already paid for.
    upserted = 0
    with get_connection(ingest_settings) as conn:
        for records in translate_batches(
            cards,
            base_url=generation_settings.generation_base_url,
            api_key=generation_settings.generation_api_key,
            model=generation_settings.generation_model,
        ):
            upsert_cards(conn, [r.__dict__ for r in records])
            upserted += len(records)
    print(f"Upserted {upserted} translated cards into Postgres.")


if __name__ == "__main__":
    main()
