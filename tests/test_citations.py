from riftbound_bot.rag.citations import citation_from_metadata, format_citations


def test_rule_citation_includes_id_and_title():
    citation = citation_from_metadata(
        {"source_type": "rule", "rule_id": "805.1.a", "title": ""}
    )
    assert citation.label == "規則 805.1.a"
    assert citation.source_url is None


def test_rule_citation_with_title():
    citation = citation_from_metadata(
        {"source_type": "rule", "rule_id": "805", "title": "疾行（Accelerate）"}
    )
    assert citation.label == "規則 805 疾行（Accelerate）"


def test_rule_citation_omits_long_inline_title():
    # Rules authored with their full body text inline after `[id]` (the
    # common case in the sample corpus) shouldn't bloat the citation label.
    citation = citation_from_metadata(
        {
            "source_type": "rule",
            "rule_id": "002",
            "title": "卡牌文字凌駕於規則文字之上。當一張卡牌的內容與規則產生根本性的衝突時，以該卡牌上的說明為準。",
        }
    )
    assert citation.label == "規則 002"


def test_card_citation_includes_source_url():
    citation = citation_from_metadata(
        {
            "source_type": "card",
            "card_id": "OGN-066",
            "name_zh": "阿璃-誘人",
            "source_url": "https://riftbound.chroniclecore.com/cards/OGN-066",
        }
    )
    assert "阿璃-誘人" in citation.label
    assert "OGN-066" in citation.label
    assert citation.as_markdown() == (
        "[卡牌《阿璃-誘人》（OGN-066）](https://riftbound.chroniclecore.com/cards/OGN-066)"
    )


def test_format_citations_deduplicates_and_orders():
    metadatas = [
        {"source_type": "rule", "rule_id": "805.1", "title": ""},
        {"source_type": "rule", "rule_id": "805.1", "title": ""},
        {"source_type": "rule", "rule_id": "809", "title": "偏斜（Deflect）"},
    ]
    formatted = format_citations(metadatas)
    lines = formatted.splitlines()
    assert len(lines) == 2
    assert lines[0] == "[來源1][來源2] 規則 805.1"
    assert lines[1] == "[來源3] 規則 809 偏斜（Deflect）"


def test_format_citations_numbers_by_retrieval_index_not_line_position():
    # The markers must track each entry's position in `metadatas` — which is
    # what the context block numbered — rather than the position of the
    # rendered line. Slots 2 and 4 collapse onto one label here, so numbering
    # the output lines 1..n would label 809 as [來源3] when the model saw it
    # at slot 3 and 811 as [來源4] when the model saw it at slot 5.
    metadatas = [
        {"source_type": "card", "card_id": "OGN-245", "name_zh": "團結之印"},
        {"source_type": "rule", "rule_id": "805.1", "title": ""},
        {"source_type": "rule", "rule_id": "809", "title": "偏斜（Deflect）"},
        {"source_type": "rule", "rule_id": "805.1", "title": ""},
        {"source_type": "rule", "rule_id": "811", "title": "待命（Hidden）"},
    ]
    lines = format_citations(metadatas).splitlines()
    assert lines == [
        "[來源1] 卡牌《團結之印》（OGN-245）",
        "[來源2][來源4] 規則 805.1",
        "[來源3] 規則 809 偏斜（Deflect）",
        "[來源5] 規則 811 待命（Hidden）",
    ]
    # Every index the model could have cited resolves to exactly one entry.
    rendered = "\n".join(lines)
    for index in range(1, len(metadatas) + 1):
        assert rendered.count(f"[來源{index}]") == 1
