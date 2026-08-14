from riftbound_bot.ingest.rules_parser import parse_rules_text


def test_parses_rule_ids_and_titles():
    # A bare `[id]` header (nothing trailing on that line) is a short heading;
    # the rule's body text goes on the following line(s).
    text = """
[805] 疾行（Accelerate）

[805.1]
疾行是一種單位（Unit）能力。

[805.1.a]
疾行的完整效果等同於：「當你打出我時...」
"""
    chunks = parse_rules_text(text, source_file="sample.md")
    assert [c.rule_id for c in chunks] == ["805", "805.1", "805.1.a"]
    assert chunks[0].title == "疾行（Accelerate）"
    assert chunks[1].body == "疾行是一種單位（Unit）能力。"
    assert chunks[0].source_file == "sample.md"


def test_inline_text_after_id_is_captured_as_title():
    # The sample corpus mostly authors rules with the full text inline after
    # `[id]` on one line — still fully captured, just under `title` not `body`.
    chunks = parse_rules_text("[002] 卡牌文字凌駕於規則文字之上。")
    assert chunks[0].title == "卡牌文字凌駕於規則文字之上。"
    assert chunks[0].body == ""
    assert chunks[0].text == "卡牌文字凌駕於規則文字之上。"


def test_multiline_body_is_joined():
    text = """
[001] 黃金規則

第一行內容。
第二行內容。
"""
    chunks = parse_rules_text(text)
    assert len(chunks) == 1
    assert chunks[0].body == "第一行內容。\n第二行內容。"


def test_html_comments_are_stripped():
    text = """
<!--
這是說明文字，不應該被解析成規則。
[999] 不應該出現
-->

[001] 真正的規則
內容
"""
    chunks = parse_rules_text(text)
    assert len(chunks) == 1
    assert chunks[0].rule_id == "001"


def test_title_only_rule_has_empty_body():
    chunks = parse_rules_text("[100] 遊戲概念\n\n[101] 牌組構築\n")
    assert chunks[0].body == ""
    assert chunks[0].title == "遊戲概念"
    assert chunks[0].text == "遊戲概念"


def test_real_sample_corpus_parses():
    """Reads the shipped seed through the package rather than a repo-relative
    path, so this no longer depends on pytest's working directory."""
    from riftbound_bot.ingest import seeds

    chunks = parse_rules_text(seeds.rules_markdown())
    ids = {c.rule_id for c in chunks}
    assert "805.1.a" in ids
    assert "809.1.c.1" in ids
    assert len(chunks) > 20
