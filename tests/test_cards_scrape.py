"""Normalization of the gallery payload, focused on the image assets.

The image path is the part most likely to break silently: it is read out of a
payload key the site added mid-project, and it is the one field whose absence
would still produce a perfectly valid-looking card record.
"""
from __future__ import annotations

from riftbound_bot.ingest.cards_scrape import normalize_card

RAW = {
    "id": "VEN-001",
    "zh_hant": {"name": "巴凱旋沙者", "effect": "{{強化5}}。費用減少{{3}}。"},
    "en": {"name": "Baccai Sandspinner"},
    "stats": {
        "category": "單位",
        "color": "red",
        "energy": 6,
        "power": 6,
        "might": None,
        "rarity": "普通",
    },
    "tags": ["恕瑞瑪"],
    "assets": {
        "img_zh_hans": "unified/VEN-001_sc.png",
        "img_en": "unified/VEN-001_en.png",
    },
}


def test_normalize_resolves_the_asset_path_to_an_absolute_url():
    card = normalize_card(RAW)

    assert card.image_url == "https://riftbound.chroniclecore.com/unified/VEN-001_sc.png"


def test_normalize_reads_the_image_path_instead_of_templating_the_card_id():
    """45 of the 1,256 cards carry a path a `unified/{id}_sc.png` template
    would get wrong — ids ending in `*` are spelled `_star_` in the filename,
    so a templated URL 404s on every one of them."""
    card = normalize_card(
        {**RAW, "id": "VEN-189*", "assets": {"img_zh_hans": "unified/VEN-189_star_sc.png"}}
    )

    assert card.image_url == (
        "https://riftbound.chroniclecore.com/unified/VEN-189_star_sc.png"
    )


def test_normalize_leaves_the_image_empty_when_the_payload_has_no_assets():
    """Card data scraped before the site published `assets` — build_client
    refuses to start on it rather than serving imageless cards."""
    card = normalize_card({k: v for k, v in RAW.items() if k != "assets"})

    assert card.image_url == ""


def test_normalize_keeps_the_chinese_face_not_the_english_one():
    """`img_en` is missing for 68 cards; `img_zh_hans` is present for all."""
    card = normalize_card(RAW)

    assert card.image_url.endswith("_sc.png")


def test_the_embedding_text_does_not_mention_the_image():
    """Adding the image to `text` would change every embedding vector and
    force a full re-index for data that has nothing to do with similarity."""
    card = normalize_card(RAW)

    assert "http" not in card.text
    assert "unified" not in card.text
