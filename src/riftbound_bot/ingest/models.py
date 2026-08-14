"""The card shapes the ingest sources produce.

`CardRecord` is what lands in Postgres, and both card sources build it — so
it lives here rather than in either of them. It previously lived in
cards_scrape, which meant the fallback source imported its core data type
from the primary one it exists to replace.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CardRecord:
    """One card as stored in the `cards` table."""

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

    def as_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ApiCard:
    """One card from the Riftcodex API, before translation.

    Same card, English text and no `source_url` yet — `to_record` supplies
    the Chinese fields and completes the shape.
    """

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

    def to_record(self, name_zh: str, text_zh: str, source_url: str) -> CardRecord:
        return CardRecord(
            id=self.id,
            set=self.set,
            collector_number=self.collector_number,
            name_zh=name_zh,
            name_en=self.name_en,
            category=self.category,
            color=self.color,
            energy=self.energy,
            power=self.power,
            might=self.might,
            rarity=self.rarity,
            tags=self.tags,
            rules_text_zh=text_zh,
            source_url=source_url,
        )
