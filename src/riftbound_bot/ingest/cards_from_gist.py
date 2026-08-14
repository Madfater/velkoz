"""Fallback card data source: OwenMelbz's community Riftbound card-data gist
(English, sourced from Riot's own CMS assets), translated via the configured
generation model.

Use this only if `cards_scrape.py` breaks (chroniclecore.com changes its
internal data shape) — the design doc treats chroniclecore.com as the
primary source since it's already native Traditional Chinese and needs no
translation. This path costs generation-model calls, one batch request per
`BATCH_SIZE` cards.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx
from langchain_core.language_models.chat_models import BaseChatModel

from riftbound_bot.ingest.cards_scrape import CardRecord
from riftbound_bot.rag.llm import build_chat_model

GIST_RAW_URL = (
    "https://gist.githubusercontent.com/OwenMelbz/"
    "e04dadf641cc9b81cb882b4612343112/raw/riftbound.json"
)
BATCH_SIZE = 20

_TAG_RE = re.compile(r"<[^>]+>")

_TRANSLATION_SYSTEM_PROMPT = """\
你是 Riftbound 集換式卡牌遊戲的繁體中文在地化人員。將下列英文卡牌名稱與規則文字翻譯成
繁體中文，用語需與官方 TCG 規則書的風格一致（簡潔、精確、避免意譯）。

保留方括號關鍵字（例如 [Accelerate]、[Deflect]）不要翻譯方括號本身的格式，但關鍵字名稱本身
請翻成對應的繁體中文術語（Accelerate -> 疾行、Deflect -> 偏斜、Action -> 行動、
Reaction -> 反應、Equip -> 裝備、Quick-Draw -> 快拔）；遇到其他未列出的關鍵字，選擇最貼近
字面意義且簡短的翻譯。

輸入是一個 JSON 陣列，每個元素是 {"id": "...", "name": "...", "text": "..."}。
輸出必須是「同樣長度、同樣順序」的 JSON 陣列，每個元素是
{"id": "...", "name_zh": "...", "text_zh": "..."}，不要添加任何其他文字或說明。
"""


@dataclass(frozen=True)
class GistCard:
    id: str
    name_en: str
    text_en: str
    set: str
    collector_number: str
    category: str
    domain: str
    rarity: str
    energy: int | None
    power: int | None


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub("", html or "")
    return re.sub(r"\s+", " ", text).strip()


def fetch_gist_cards() -> list[GistCard]:
    resp = httpx.get(GIST_RAW_URL, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    raw_cards = resp.json()

    cards = []
    for raw in raw_cards:
        card_types = raw.get("cardType") or []
        domains = raw.get("domains") or []
        rarity = raw.get("rarity") or {}
        cards.append(
            GistCard(
                id=raw["set"] + "-" + str(raw["collectorNumber"]).zfill(3),
                name_en=raw.get("name", ""),
                text_en=_strip_html(raw.get("text", "")),
                set=raw.get("set", ""),
                collector_number=str(raw.get("collectorNumber", "")).zfill(3),
                category=card_types[0]["label"] if card_types else "",
                domain=domains[0]["label"] if domains else "",
                rarity=rarity.get("label", ""),
                energy=raw.get("energy"),
                power=raw.get("power"),
            )
        )
    return cards


def _translate_batch(llm: BaseChatModel, batch: list[GistCard]) -> dict[str, tuple[str, str]]:
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
    cards: list[GistCard], base_url: str, api_key: str, model: str
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
                    color=card.domain,
                    energy=card.energy,
                    power=card.power,
                    might=None,
                    rarity=card.rarity,
                    tags=[],
                    rules_text_zh=text_zh,
                    source_url="",
                )
            )
    return records


def main() -> None:
    import sys

    from riftbound_bot.config import Settings

    settings = Settings.load()
    cards = fetch_gist_cards()
    records = translate_all(
        cards,
        base_url=settings.generation_base_url,
        api_key=settings.generation_api_key,
        model=settings.generation_model,
    )
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/cards/cards.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump([r.__dict__ for r in records], fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} translated cards to {out_path}")


if __name__ == "__main__":
    main()
