"""The card catalog behind /card: a name-searchable view of the `cards` table.

This is the first thing in the bot to read `cards` at request time — every
other runtime query goes to `embeddings`. It reads the source table rather
than the index because the index deliberately carries only what retrieval
needs (id, both names, rarity, source url); energy/power/might/color/tags and
the image never make it into embedding metadata, and widening that metadata to
carry them would force a full re-embed of the corpus for data that has nothing
to do with similarity search.

The whole table is loaded once at startup and searched in memory. At ~1,256
rows that is a fraction of a megabyte, and it is what Discord's autocomplete
contract requires anyway: a keystroke-rate callback with a hard 3-second
budget cannot afford a round trip per character.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import structlog
from psycopg_pool import ConnectionPool

from riftbound_bot.ingest.db import CARDS_TABLE

logger = structlog.get_logger("riftbound_bot.cards")

# Discord's own ceilings, not ours: a command may offer at most 25 autocomplete
# choices, and a choice's name may not exceed 100 characters.
MAX_AUTOCOMPLETE_CHOICES = 25
MAX_CHOICE_LABEL = 100

# Alternate-art reprints duplicate a card's name with the same rules text, so
# they lose ties when a query matches a name shared by several printings.
ALT_ART_RARITY = "異畫"


@dataclass(frozen=True)
class Card:
    """One printed card, as stored in `cards.data`."""

    id: str
    name_zh: str
    name_en: str
    category: str
    color: str
    rarity: str
    rules_text_zh: str
    source_url: str
    image_url: str
    set: str = ""
    collector_number: str = ""
    energy: int | None = None
    power: int | None = None
    might: int | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, data: dict) -> Card:
        """Builds a Card from a `cards.data` payload, ignoring unknown keys.

        Tolerant of extra keys on purpose: `data` is whatever the scraper last
        wrote, and an upstream schema addition should not crash the bot at
        startup.
        """
        return cls(
            id=str(data.get("id") or ""),
            name_zh=str(data.get("name_zh") or ""),
            name_en=str(data.get("name_en") or ""),
            category=str(data.get("category") or ""),
            color=str(data.get("color") or ""),
            rarity=str(data.get("rarity") or ""),
            rules_text_zh=str(data.get("rules_text_zh") or ""),
            source_url=str(data.get("source_url") or ""),
            image_url=str(data.get("image_url") or ""),
            set=str(data.get("set") or ""),
            collector_number=str(data.get("collector_number") or ""),
            energy=data.get("energy"),
            power=data.get("power"),
            might=data.get("might"),
            tags=list(data.get("tags") or []),
        )

    @property
    def display_name(self) -> str:
        """Chinese name with the English one alongside, where there is one."""
        return f"{self.name_zh}（{self.name_en}）" if self.name_en else self.name_zh

    def choice_label(self) -> str:
        """Autocomplete row text, always ending in the card id.

        The id is not decoration: 188 of the Chinese names are shared by more
        than one card (每個「狂怒符文」在十個系列裡各印了一張), so a name-only
        row would leave the user picking blind between identical-looking
        options.
        """
        label = f"{self.display_name} · {self.id}"
        if len(label) <= MAX_CHOICE_LABEL:
            return label
        # Trim the name, never the id — the id is what makes the row unique.
        keep = MAX_CHOICE_LABEL - len(f"… · {self.id}")
        return f"{self.display_name[:keep]}… · {self.id}"


class CardSource(Protocol):
    """The one call the catalog makes against storage.

    Narrow for the same reason RetrievalStore is (see rag/vectorstore.py): it
    keeps the real dependency visible, and lets tests hand over a list of dicts
    without a database.
    """

    def fetch_all_cards(self) -> list[dict]: ...


class PgCardSource:
    """Reads every row of `cards`, once, through the bot's existing pool."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def fetch_all_cards(self) -> list[dict]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT data FROM {CARDS_TABLE}")
            return [row[0] for row in cur.fetchall()]


def _sort_key(card: Card, query: str) -> tuple[bool, int, bool, str]:
    """Ranks one match: prefix hits first, then shorter names, then base art.

    Prefix before substring is what makes typing feel like completion rather
    than grep —「符文」should offer 符文牆 before 狂怒符文, because the former is
    what the user is part-way through typing.
    """
    name_zh = card.name_zh.casefold()
    name_en = card.name_en.casefold()
    starts = name_zh.startswith(query) or name_en.startswith(query)
    shortest = min(
        (len(name) for name in (name_zh, name_en) if name and query in name),
        default=len(name_zh),
    )
    return (not starts, shortest, card.rarity == ALT_ART_RARITY, card.id)


class CardCatalog:
    """Every card, in memory, searchable by Chinese name, English name, or id."""

    def __init__(self, cards: list[Card]) -> None:
        self._cards = cards
        self._by_id = {card.id: card for card in cards}

    @classmethod
    def from_source(cls, source: CardSource) -> CardCatalog:
        catalog = cls([Card.from_row(row) for row in source.fetch_all_cards()])
        logger.info(
            "cards.catalog_loaded",
            cards=len(catalog),
            without_image=catalog.cards_missing_images,
        )
        return catalog

    def __len__(self) -> int:
        return len(self._cards)

    @property
    def cards_missing_images(self) -> int:
        return sum(1 for card in self._cards if not card.image_url)

    def get(self, card_id: str) -> Card | None:
        return self._by_id.get(card_id)

    def search(self, query: str, limit: int = MAX_AUTOCOMPLETE_CHOICES) -> list[Card]:
        """Cards whose Chinese name, English name, or id contains `query`.

        Case-insensitive so English names can be typed in lower case, and
        substring rather than prefix-only so 「符文」 finds 狂怒符文 — a Chinese
        card name is routinely typed from the middle, where the distinctive
        word sits.

        An empty query returns the head of the catalog instead of nothing, so
        the autocomplete menu is populated the moment the command is opened.
        """
        normalized = query.strip().casefold()
        if not normalized:
            return self._cards[:limit]
        matches = [
            card
            for card in self._cards
            if normalized in card.name_zh.casefold()
            or (card.name_en and normalized in card.name_en.casefold())
            or normalized in card.id.casefold()
        ]
        matches.sort(key=lambda card: _sort_key(card, normalized))
        return matches[:limit]

    def resolve(self, query: str) -> Card | None:
        """The single card a submitted /card argument refers to.

        Handles both ways the argument can arrive. Picking a row from the
        autocomplete menu submits that card's id, which hits `get` and is
        exact. Typing a name and pressing enter without picking anything
        submits raw text, which falls through to a search — so the command
        still answers instead of scolding the user for skipping the menu.
        """
        exact = self.get(query.strip())
        if exact is not None:
            return exact
        matches = self.search(query, limit=1)
        return matches[0] if matches else None
