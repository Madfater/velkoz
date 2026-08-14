"""Covers document construction, which decides what retrieval can ever find.

Both builders are pure functions over the rows Postgres holds, so these run
without a database or an embedding endpoint.
"""
from riftbound_bot.ingest.build_index import (
    _card_documents_from_dicts,
    _card_embed_text,
    _rule_documents_from_chunks,
)
from riftbound_bot.ingest.rules_parser import RuleChunk


def _chunk(rule_id: str, title: str, body: str = "") -> RuleChunk:
    return RuleChunk(rule_id=rule_id, title=title, body=body, source_file="core.md")


def _by_id(documents) -> dict[str, str]:
    return {doc.metadata["rule_id"]: doc.page_content for doc in documents}


def test_a_sub_rule_is_embedded_with_its_headings():
    # 805.1 is "疾行是一種單位能力。" — without its section name attached the
    # word Accelerate never appears in its own text, and it ranked 1306th of
    # 1307 for a question asking exactly what 疾行 is.
    documents = _rule_documents_from_chunks([
        _chunk("805", "疾行（Accelerate）"),
        _chunk("805.1", "疾行是一種單位（Unit）能力。"),
    ])

    assert _by_id(documents)["805.1"] == "疾行（Accelerate）\n疾行是一種單位（Unit）能力。"


def test_the_whole_ancestor_chain_is_prepended_outermost_first():
    # 135.2.e.5.a's useful context is its immediate parent (135.2.e 符號),
    # not just the broad top-level section.
    documents = _rule_documents_from_chunks([
        _chunk("135", "規則文字（Rules Text）"),
        _chunk("135.2", "符號與縮寫"),
        _chunk("135.2.e", "符號"),
        _chunk("135.2.e.5.a", "[A] 可以任意屬性的力量支付。"),
    ])

    assert _by_id(documents)["135.2.e.5.a"] == (
        "規則文字（Rules Text）\n符號與縮寫\n符號\n[A] 可以任意屬性的力量支付。"
    )


def test_ancestors_that_are_rule_statements_are_not_prepended():
    # Every rule is written as `[id] text` on one line, so "has a body" can't
    # separate headings from rule statements — a terminal 。／！／？ can.
    # Dragging a parent's full sentence into each descendant would bury the
    # descendant's own text, sometimes under a lengthy worked example.
    documents = _rule_documents_from_chunks([
        _chunk("810", "單位可以在戰場上進行攻擊，並且會造成傷害。"),
        _chunk("810.1", "攻擊時橫置該單位。"),
    ])

    assert _by_id(documents)["810.1"] == "攻擊時橫置該單位。"


def test_a_heading_ending_in_a_gloss_paren_still_counts_as_a_heading():
    documents = _rule_documents_from_chunks([
        _chunk("809", "偏斜（Deflect）"),
        _chunk("809.1", "偏斜可以取消一次傷害。"),
    ])

    assert _by_id(documents)["809.1"].startswith("偏斜（Deflect）\n")


def test_rules_with_no_text_are_dropped():
    documents = _rule_documents_from_chunks([_chunk("900", ""), _chunk("901", "有內容。")])

    assert list(_by_id(documents)) == ["901"]


def test_no_rules_at_all_is_not_an_error():
    assert _rule_documents_from_chunks([]) == []


def _card(**overrides) -> dict:
    card = {
        "id": "OGN-121",
        "name_zh": "魅惑之靈",
        "name_en": "Bewitching Spirit",
        "category": "單位",
        "color": "purple",
        "rarity": "普通",
        "energy": 3,
        "power": None,
        "might": 2,
        "tags": ["靈體", "暗影島"],
        "rules_text_zh": "當我進場時，抽 1 張牌。",
        "source_url": "https://example.test/cards/OGN-121",
    }
    card.update(overrides)
    return card


def test_card_text_carries_the_attributes_questions_are_asked_about():
    # Colour, rarity, stats and tags were absent from the embedded string, so
    # "哪些卡可以獲得黃色力量"-shaped questions had nothing to match on.
    text = _card_embed_text(_card())

    for expected in ("魅惑之靈", "OGN-121", "單位", "purple", "普通", "費用 3", "戰力 2", "靈體"):
        assert expected in text
    assert text.endswith("效果：當我進場時，抽 1 張牌。")


def test_stats_a_card_does_not_have_are_left_out():
    # This card has no power, so no empty "力量" label should appear.
    assert "力量" not in _card_embed_text(_card(power=None))
    assert "力量 4" in _card_embed_text(_card(power=4))


def test_a_card_without_rules_text_is_still_indexed():
    # Vanilla units have no effect text; they are still retrievable cards.
    text = _card_embed_text(_card(rules_text_zh=""))

    assert "魅惑之靈" in text
    assert "效果：" not in text


def test_card_metadata_carries_what_citations_and_filtering_need():
    doc = _card_documents_from_dicts([_card()])[0]

    assert doc.metadata["card_id"] == "OGN-121"
    assert doc.metadata["name_zh"] == "魅惑之靈"
    assert doc.metadata["rarity"] == "普通"
    assert doc.metadata["source_url"].endswith("OGN-121")


def test_a_malformed_card_row_is_skipped_not_fatal():
    # One bad row used to KeyError and abort the entire rebuild.
    documents = _card_documents_from_dicts([{"id": "OGN-999"}, _card()])

    assert [doc.metadata["card_id"] for doc in documents] == ["OGN-121"]
