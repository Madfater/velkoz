"""Covers the primary card source's parsing of the gallery page.

`cards_scrape` reads the whole card dataset out of the RSC payload Next.js
embeds in the page's `self.__next_f.push(...)` calls, so these tests build
that structure by hand rather than hitting the network.
"""
import json

import pytest

from riftbound_bot.ingest.cards_scrape import (
    _clean_effect_text,
    _extract_card_dicts,
    normalize_card,
)


def _raw_card(card_id: str = "OGN-121", name_zh: str = "魅惑之靈") -> dict:
    return {
        "id": card_id,
        "zh_hant": {"name": name_zh, "effect": "當我進場時，{{抽}} 1 張牌。"},
        "en": {"name": "Bewitching Spirit"},
        "stats": {
            "category": "單位",
            "color": "purple",
            "rarity": "普通",
            "energy": 3,
            "power": None,
            "might": 2,
        },
        "tags": ["靈體"],
    }


def _push_chunk(payload: str) -> str:
    """One `self.__next_f.push([1, "..."])` call, JSON-escaped as the page has it."""
    return f"self.__next_f.push([1,{json.dumps(payload)}])"


def _gallery_html(cards: list[dict], *, segment_id: str = "a1") -> str:
    # The card array arrives as a numbered RSC segment, preceded by unrelated
    # ones that are not JSON at all.
    card_segment = json.dumps({"cards": cards}, ensure_ascii=False)
    segments = f'0:"unrelated preamble"\n{segment_id}:{card_segment}\n'
    return f"<html><script>{_push_chunk(segments)}</script></html>"


def test_card_array_is_found_by_shape_in_the_rsc_payload():
    cards = [_raw_card(f"OGN-{i:03d}") for i in range(60)]

    found = _extract_card_dicts(_gallery_html(cards))

    assert len(found) == 60
    assert found[0]["id"] == "OGN-000"


def test_card_array_split_across_push_chunks_is_reassembled():
    # Next.js streams one logical segment across several push calls, so the
    # chunks have to be concatenated before anything is decoded.
    cards = [_raw_card(f"OGN-{i:03d}") for i in range(60)]
    segments = f'a1:{json.dumps({"cards": cards}, ensure_ascii=False)}\n'
    half = len(segments) // 2
    html = f"<script>{_push_chunk(segments[:half])}{_push_chunk(segments[half:])}</script>"

    assert len(_extract_card_dicts(html)) == 60


def test_a_page_without_card_data_reports_the_fallback():
    # The operator needs to be told the site shape changed, and where to go.
    chunk = _push_chunk('0:"nothing useful here"')
    html = f"<script>{chunk}</script>"

    with pytest.raises(RuntimeError, match="cards_from_api"):
        _extract_card_dicts(html)


def test_undecodable_push_chunks_report_the_fallback_too():
    # Previously a raw JSONDecodeError escaped from here instead.
    with pytest.raises(RuntimeError, match="cards_from_api"):
        _extract_card_dicts('<script>self.__next_f.push([1,"unterminated])</script>')


def test_normalize_splits_the_id_and_strips_keyword_markup():
    card = normalize_card(_raw_card())

    assert (card.id, card.set, card.collector_number) == ("OGN-121", "OGN", "121")
    assert card.rules_text_zh == "當我進場時，抽 1 張牌。"
    assert card.source_url.endswith("/cards/OGN-121")


def test_keyword_markup_is_reduced_to_its_text():
    assert _clean_effect_text("{{疾行}}：造成 {{2}} 點傷害") == "疾行：造成 2 點傷害"
