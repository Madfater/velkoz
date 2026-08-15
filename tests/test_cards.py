"""Catalog behaviour behind /card — name search, ranking, and resolution.

The search rules here are the whole user experience of the command: the
autocomplete menu is the only thing standing between a user and the 188
Chinese card names that more than one card shares.
"""
from __future__ import annotations

import pytest

from riftbound_bot.bot import _load_catalog
from riftbound_bot.cards import Card, CardCatalog


def make_card(
    card_id: str,
    name_zh: str,
    name_en: str = "",
    rarity: str = "普通",
    **overrides,
) -> Card:
    data = {
        "id": card_id,
        "name_zh": name_zh,
        "name_en": name_en,
        "category": "單位",
        "color": "red",
        "rarity": rarity,
        "rules_text_zh": "效果",
        "source_url": f"https://riftbound.chroniclecore.com/cards/{card_id}",
        "image_url": f"https://riftbound.chroniclecore.com/unified/{card_id}_sc.png",
        "set": card_id.split("-")[0],
        "collector_number": card_id.split("-")[-1],
        "energy": 3,
        "power": 2,
        "might": None,
        "tags": ["恕瑞瑪"],
    }
    data.update(overrides)
    return Card.from_row(data)


class FakeCardSource:
    def __init__(self, rows):
        self._rows = rows

    def fetch_all_cards(self):
        return self._rows


def test_from_row_ignores_keys_the_bot_does_not_know_about():
    """An upstream schema addition should not crash the bot at startup."""
    card = Card.from_row(
        {"id": "VEN-001", "name_zh": "巴凱旋沙者", "brand_new_upstream_field": 1}
    )

    assert card.id == "VEN-001"
    assert card.name_zh == "巴凱旋沙者"


def test_from_row_defaults_missing_stats_to_none_rather_than_zero():
    """A spell has no power; rendering it as 0 would state something false."""
    card = Card.from_row({"id": "VEN-002", "name_zh": "法術"})

    assert card.energy is None
    assert card.power is None
    assert card.might is None
    assert card.tags == []


def test_search_matches_chinese_and_english_names_in_one_query():
    catalog = CardCatalog([make_card("VEN-001", "巴凱旋沙者", "Baccai Sandspinner")])

    assert [c.id for c in catalog.search("巴凱")] == ["VEN-001"]
    assert [c.id for c in catalog.search("Sandspinner")] == ["VEN-001"]


def test_search_of_english_names_is_case_insensitive():
    catalog = CardCatalog([make_card("VEN-001", "巴凱旋沙者", "Baccai Sandspinner")])

    assert [c.id for c in catalog.search("sandspinner")] == ["VEN-001"]


def test_search_matches_a_card_id_so_a_known_code_can_be_pasted_in():
    catalog = CardCatalog([make_card("VEN-001", "巴凱旋沙者", "Baccai Sandspinner")])

    assert [c.id for c in catalog.search("ven-001")] == ["VEN-001"]


def test_search_matches_the_middle_of_a_chinese_name():
    """Chinese card names are routinely typed from the distinctive word in the
    middle rather than the first character."""
    catalog = CardCatalog([make_card("OGN-001", "狂怒符文")])

    assert [c.id for c in catalog.search("符文")] == ["OGN-001"]


def test_search_ranks_prefix_matches_above_substring_matches():
    catalog = CardCatalog(
        [make_card("OGN-001", "狂怒符文"), make_card("OGN-002", "符文牆")]
    )

    assert [c.id for c in catalog.search("符文")] == ["OGN-002", "OGN-001"]


def test_search_ranks_the_base_printing_above_its_alternate_art():
    """Alternate art duplicates the name with identical rules text, so it is
    the less useful of the two when a query matches both."""
    catalog = CardCatalog(
        [
            make_card("UNL-224", "秘術普羅", rarity="異畫"),
            make_card("UNL-100", "秘術普羅", rarity="普通"),
        ]
    )

    assert [c.id for c in catalog.search("秘術普羅")] == ["UNL-100", "UNL-224"]


def test_search_caps_results_at_the_requested_limit():
    """Discord rejects an autocomplete response carrying more than 25 choices."""
    catalog = CardCatalog([make_card(f"OGN-{i:03d}", f"符文{i}") for i in range(40)])

    assert len(catalog.search("符文")) == 25
    assert len(catalog.search("符文", limit=5)) == 5


def test_search_with_an_empty_query_offers_the_head_of_the_catalog():
    """So the autocomplete menu is populated the moment the command opens,
    rather than staying blank until the first keystroke."""
    catalog = CardCatalog([make_card("OGN-001", "甲"), make_card("OGN-002", "乙")])

    assert [c.id for c in catalog.search("")] == ["OGN-001", "OGN-002"]


def test_search_returns_nothing_for_a_name_no_card_has():
    catalog = CardCatalog([make_card("OGN-001", "狂怒符文")])

    assert catalog.search("青眼白龍") == []


def test_resolve_prefers_an_exact_card_id_over_a_name_search():
    """Picking an autocomplete row submits the id, and that choice must win —
    it is the disambiguation the user just performed."""
    catalog = CardCatalog(
        [make_card("UNL-100", "秘術普羅"), make_card("UNL-224", "秘術普羅")]
    )

    assert catalog.resolve("UNL-224").id == "UNL-224"


def test_resolve_falls_back_to_search_for_free_text():
    """Typing a name and pressing enter without picking a row still answers."""
    catalog = CardCatalog([make_card("VEN-001", "巴凱旋沙者", "Baccai Sandspinner")])

    assert catalog.resolve("Baccai").id == "VEN-001"


def test_resolve_returns_none_when_nothing_matches():
    catalog = CardCatalog([make_card("VEN-001", "巴凱旋沙者")])

    assert catalog.resolve("青眼白龍") is None


def test_choice_label_carries_the_card_id_to_separate_shared_names():
    card = make_card("VEN-001", "巴凱旋沙者", "Baccai Sandspinner")

    assert card.choice_label() == "巴凱旋沙者（Baccai Sandspinner） · VEN-001"


def test_choice_label_keeps_the_id_when_it_has_to_truncate():
    """Discord caps a choice label at 100 characters; the id is the half that
    makes the row unique, so the name is what gets trimmed."""
    card = make_card("VEN-001", "長" * 90, "L" * 90)

    label = card.choice_label()

    assert len(label) <= 100
    assert label.endswith("· VEN-001")


def test_catalog_reports_cards_that_have_no_image():
    """build_client refuses to start on card data that predates image capture,
    and this count is what it checks."""
    catalog = CardCatalog(
        [make_card("VEN-001", "甲"), make_card("VEN-002", "乙", image_url="")]
    )

    assert catalog.cards_missing_images == 1


def test_load_catalog_refuses_data_where_no_card_has_an_image():
    """That is the signature of card rows written before image capture, and
    the case bootstrap's backfill repairs."""
    source = FakeCardSource([{"id": "VEN-001", "name_zh": "甲"}])

    with pytest.raises(RuntimeError, match="predates image capture"):
        _load_catalog(source)


def test_load_catalog_refuses_an_empty_cards_table():
    with pytest.raises(RuntimeError, match="No cards in Postgres"):
        _load_catalog(FakeCardSource([]))


def test_load_catalog_serves_data_where_only_some_images_are_missing():
    """Upstream missing art for a few cards is not a reason to take /ask down
    with /card — those render without an image."""
    source = FakeCardSource(
        [
            {"id": "VEN-001", "name_zh": "甲", "image_url": "https://x/1.png"},
            {"id": "VEN-002", "name_zh": "乙"},
        ]
    )

    catalog = _load_catalog(source)

    assert len(catalog) == 2
    assert catalog.cards_missing_images == 1


def test_from_source_builds_the_catalog_from_stored_rows():
    source = FakeCardSource([{"id": "VEN-001", "name_zh": "巴凱旋沙者"}])

    catalog = CardCatalog.from_source(source)

    assert len(catalog) == 1
    assert catalog.get("VEN-001").name_zh == "巴凱旋沙者"
