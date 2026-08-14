from dataclasses import dataclass

from riftbound_bot.rag.chain import (
    MAX_EXACT_MATCH_NAMES,
    NO_CONTEXT_REPLY,
    RiftboundRagChain,
)
from riftbound_bot.rag.lexical_index import rule_sort_key


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict


@dataclass
class FakeResponse:
    content: str


class FakeVectorstore:
    """Keyed by source_type so tests can control each pool independently,
    mirroring how the real TurboQuantVectorStore filter={"source_type": ...}
    search works. `cards`/`rules` back similarity_search's bulk-fetch calls
    (large k, no scores) that RiftboundRagChain uses to load its exact-match
    indexes at construction time — mirrors the real store's "return every
    matching document once k exceeds the candidate count" behavior.
    """

    def __init__(
        self,
        by_type: dict[str, list[tuple[FakeDoc, float]]],
        cards: list[FakeDoc] | None = None,
        rules: list[FakeDoc] | None = None,
    ):
        self._by_type = by_type
        self._cards = cards or []
        self._rules = rules or []
        self.calls: list[tuple[str, int, dict]] = []
        self.bulk_fetch_calls: list[dict] = []

    def similarity_search_with_relevance_scores(self, query, k, filter):
        self.calls.append((query, k, filter))
        return self._by_type.get(filter["source_type"], [])

    def similarity_search(self, query, k, filter):
        self.bulk_fetch_calls.append(filter)
        return self._rules if filter.get("source_type") == "rule" else self._cards


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.last_messages: list | None = None

    def invoke(self, messages):
        self.last_messages = messages
        return FakeResponse(content=self.reply)


def make_rule_doc(rule_id: str, text: str) -> FakeDoc:
    return FakeDoc(page_content=text, metadata={"source_type": "rule", "rule_id": rule_id, "title": ""})


def make_card_doc(card_id: str, name_zh: str, text: str, rarity: str = "普通") -> FakeDoc:
    return FakeDoc(
        page_content=text,
        metadata={
            "source_type": "card",
            "card_id": card_id,
            "name_zh": name_zh,
            "source_url": "",
            "rarity": rarity,
        },
    )


def make_keyword_header_doc(rule_id: str, term_zh: str, term_en: str) -> FakeDoc:
    title = f"{term_zh}（{term_en}）"
    return FakeDoc(page_content=title, metadata={"source_type": "rule", "rule_id": rule_id, "title": title})


def test_ask_returns_no_context_reply_when_nothing_relevant():
    chain = RiftboundRagChain(
        vectorstore=FakeVectorstore({}),
        llm=FakeLLM("should not be called"),
        pool_per_type=10,
        k=6,
        score_threshold=0.45,
    )
    result = chain.ask("什麼是疾行？")
    assert result.answer == NO_CONTEXT_REPLY
    assert result.citations_markdown == ""


def test_ask_merges_rules_and_cards_pools_by_score():
    vectorstore = FakeVectorstore(
        {
            "rule": [(make_rule_doc("805.1", "疾行是一種單位能力。"), 0.9)],
            "card": [(make_card_doc("OGN-001", "測試卡", "測試卡效果文字。"), 0.6)],
        }
    )
    llm = FakeLLM("疾行是……[1]")
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=llm, pool_per_type=10, k=6, score_threshold=0.45)

    result = chain.ask("什麼是疾行？")

    assert result.answer == "疾行是……[1]"
    labels = result.citations_markdown.splitlines()
    assert labels[0] == "[1] 規則 805.1"  # higher score, comes first
    assert "測試卡" in labels[1]


def test_ask_filters_each_pool_by_score_threshold_independently():
    vectorstore = FakeVectorstore(
        {
            "rule": [(make_rule_doc("805.1", "疾行是一種單位能力。"), 0.9)],
            "card": [(make_card_doc("OGN-001", "不相關卡", "不相關內容"), 0.1)],
        }
    )
    llm = FakeLLM("疾行是……[1]")
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=llm, pool_per_type=10, k=6, score_threshold=0.45)

    result = chain.ask("什麼是疾行？")

    assert "規則 805.1" in result.citations_markdown
    assert "不相關卡" not in result.citations_markdown


def test_ask_caps_merged_pool_at_k_keeping_highest_scores():
    # Rule scores: 0.90, 0.89, 0.88, 0.87, 0.86. Card scores: 0.80, ..., 0.76.
    rules = [(make_rule_doc(str(i), f"規則內容 {i}"), round(0.9 - i * 0.01, 2)) for i in range(5)]
    cards = [(make_card_doc(f"C{i}", f"卡 {i}", f"卡牌內容 {i}"), round(0.8 - i * 0.01, 2)) for i in range(5)]
    vectorstore = FakeVectorstore({"rule": rules, "card": cards})
    chain = RiftboundRagChain(
        vectorstore=vectorstore, llm=FakeLLM("回答"), pool_per_type=10, k=3, score_threshold=0.0
    )

    retrieved = chain._retrieve("問題")

    assert [doc.metadata["rule_id"] for doc, _ in retrieved] == ["0", "1", "2"]
    assert [score for _, score in retrieved] == [0.9, 0.89, 0.88]


def test_ask_includes_history_and_context_in_prompt():
    vectorstore = FakeVectorstore({"rule": [(make_rule_doc("805.1", "疾行是一種單位能力。"), 0.9)], "card": []})
    llm = FakeLLM("回答")
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=llm, pool_per_type=10, k=6, score_threshold=0.45)

    history = [("human", "先前的問題"), ("ai", "先前的回答")]
    chain.ask("後續問題", history=history)

    messages = llm.last_messages
    assert messages[0][0] == "system"
    assert messages[1] == ("human", "先前的問題")
    assert messages[2] == ("ai", "先前的回答")
    assert messages[-1][0] == "human"
    assert "疾行是一種單位能力。" in messages[-1][1]
    assert "後續問題" in messages[-1][1]


def test_ask_queries_both_pools_with_pool_per_type_as_k():
    vectorstore = FakeVectorstore({"rule": [], "card": []})
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=15, k=6, score_threshold=0.45)
    chain.ask("問題")
    source_types = {call[2]["source_type"] for call in vectorstore.calls}
    assert source_types == {"rule", "card"}
    assert all(call[1] == 15 for call in vectorstore.calls)


def test_exact_card_name_match_is_force_included_even_below_threshold():
    # The card never scores above threshold in similarity search (mirrors the
    # confirmed real-world bug: an exact-name query still ranking a card
    # outside the top of its own pool), but its name is typed verbatim in the
    # question, so it must still surface.
    card = make_card_doc("OGN-245", "團結之印", "團結之印（Seal of Unity）效果：橫置：反應—獲得黃色。")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, cards=[card])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM("可以[1]"), pool_per_type=10, k=6, score_threshold=0.45)

    result = chain.ask("團結之印可以支付待命的費用嗎")

    assert "團結之印" in result.citations_markdown
    assert "OGN-245" in result.citations_markdown


def test_exact_matches_are_kept_even_when_they_exceed_k():
    # k bounds the similarity pool, not the exact matches: a card whose full
    # name is in the question is a deterministic lexical fact, and dropping
    # one to respect k would answer about some of the cards asked about.
    # MAX_EXACT_MATCH_NAMES is what bounds this side.
    cards = [make_card_doc(f"C{i}", f"卡牌{i}號", f"卡{i}效果") for i in range(3)]
    question = "".join(c.metadata["name_zh"] for c in cards)
    vectorstore = FakeVectorstore({"rule": [], "card": []}, cards=cards)
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=2, score_threshold=0.45)

    retrieved = chain._retrieve(question)

    assert len(retrieved) == 3


def test_exact_matches_are_capped_at_max_exact_match_names():
    cards = [make_card_doc(f"C{i}", f"獨特卡牌{i}號", f"卡{i}效果") for i in range(MAX_EXACT_MATCH_NAMES + 2)]
    question = "".join(c.metadata["name_zh"] for c in cards)
    vectorstore = FakeVectorstore({"rule": [], "card": []}, cards=cards)
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=20, score_threshold=0.45)

    retrieved = chain._retrieve(question)

    assert len(retrieved) == MAX_EXACT_MATCH_NAMES


def test_exact_match_multi_printing_prefers_non_alt_art():
    # The alt-art printing deliberately holds the lexically *lower* card_id:
    # with the ids the other way round the id tie-break alone produced the
    # expected answer, so the test passed with the rarity filter deleted.
    alt = make_card_doc("AAA-001", "團結之印", "異畫版效果文字", rarity="異畫")
    standard = make_card_doc("OGN-245", "團結之印", "標準版效果文字", rarity="史詩")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, cards=[alt, standard])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=6, score_threshold=0.45)

    retrieved = chain._retrieve("團結之印是什麼")

    assert [doc.metadata["card_id"] for doc, _ in retrieved] == ["OGN-245"]


def test_exact_match_drops_subsumed_shorter_name():
    short = make_card_doc("C1", "提摩", "提摩基礎效果")
    longer = make_card_doc("C2", "提摩-戰略家", "提摩-戰略家效果")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, cards=[short, longer])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=6, score_threshold=0.45)

    retrieved = chain._retrieve("提摩-戰略家的效果是什麼")

    assert [doc.metadata["card_id"] for doc, _ in retrieved] == ["C2"]


def test_exact_match_dedups_against_similarity_pool_hit():
    card = make_card_doc("OGN-245", "團結之印", "團結之印效果文字")
    vectorstore = FakeVectorstore({"rule": [], "card": [(card, 0.9)]}, cards=[card])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM("答案[1]"), pool_per_type=10, k=6, score_threshold=0.45)

    result = chain.ask("團結之印是什麼")

    assert result.citations_markdown.count("團結之印") == 1


def test_no_exact_match_leaves_score_sort_behavior_intact():
    rules = [(make_rule_doc(str(i), f"規則內容 {i}"), round(0.9 - i * 0.01, 2)) for i in range(5)]
    cards = [(make_card_doc(f"C{i}", f"卡 {i}", f"卡牌內容 {i}"), round(0.8 - i * 0.01, 2)) for i in range(5)]
    vectorstore = FakeVectorstore({"rule": rules, "card": cards}, cards=[])
    chain = RiftboundRagChain(
        vectorstore=vectorstore, llm=FakeLLM("回答"), pool_per_type=10, k=3, score_threshold=0.0
    )

    retrieved = chain._retrieve("問題")

    assert [doc.metadata["rule_id"] for doc, _ in retrieved] == ["0", "1", "2"]
    assert [score for _, score in retrieved] == [0.9, 0.89, 0.88]


def test_exact_keyword_match_is_force_included_even_below_threshold():
    header = make_keyword_header_doc("811", "待命", "Hidden")
    sub_rule = make_rule_doc("811.1.b", "待命的完整效果等同於：「...你可以支付 [A]...」")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, rules=[header, sub_rule])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM("可以[1]"), pool_per_type=10, k=6, score_threshold=0.45)

    result = chain.ask("團結之印可以支付待命的費用嗎")

    assert "待命" in result.citations_markdown
    assert "811" in result.citations_markdown


def test_exact_keyword_match_combines_full_subtree_into_one_document():
    header = make_keyword_header_doc("811", "待命", "Hidden")
    sub_a = make_rule_doc("811.1", "待命是一種關鍵字。")
    sub_b = make_rule_doc("811.1.b", "待命的完整效果等同於：「...你可以支付 [A]...」")
    unrelated = make_keyword_header_doc("805", "疾行", "Accelerate")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, rules=[header, sub_a, sub_b, unrelated])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=6, score_threshold=0.45)

    retrieved = chain._retrieve("待命是什麼")

    assert len(retrieved) == 1
    doc, _score = retrieved[0]
    assert doc.metadata["rule_id"] == "811"
    assert "待命是一種關鍵字" in doc.page_content
    assert "可以支付 [A]" in doc.page_content


def test_exact_keyword_match_dedups_against_similarity_pool_hit_in_subtree():
    header = make_keyword_header_doc("811", "待命", "Hidden")
    sub_rule = make_rule_doc("811.1.b", "待命的完整效果等同於：「...」")
    vectorstore = FakeVectorstore(
        {"rule": [(sub_rule, 0.9)], "card": []}, rules=[header, sub_rule]
    )
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM("答案[1]"), pool_per_type=10, k=6, score_threshold=0.45)

    result = chain.ask("待命是什麼")

    assert result.citations_markdown.count("待命") == 1


def test_no_keyword_match_leaves_score_sort_behavior_intact():
    header = make_keyword_header_doc("811", "待命", "Hidden")
    rules = [(make_rule_doc(str(i), f"規則內容 {i}"), round(0.9 - i * 0.01, 2)) for i in range(5)]
    vectorstore = FakeVectorstore({"rule": rules, "card": []}, rules=[header])
    chain = RiftboundRagChain(
        vectorstore=vectorstore, llm=FakeLLM("回答"), pool_per_type=10, k=3, score_threshold=0.0
    )

    retrieved = chain._retrieve("問題")

    assert [doc.metadata["rule_id"] for doc, _ in retrieved] == ["0", "1", "2"]


def test_symbol_expansion_pulls_in_definition_referenced_by_exact_match():
    header = make_keyword_header_doc("811", "待命", "Hidden")
    sub_rule = make_rule_doc("811.1.b", "待命的完整效果等同於：「你可以支付 [A]...」")
    symbol_def = make_rule_doc("135.2.e.5", "任意屬性的力量，以旋轉的彩虹符號表示，簡稱為 [A]。")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, rules=[header, sub_rule, symbol_def])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM("答案[1][2]"), pool_per_type=10, k=6, score_threshold=0.45)

    result = chain.ask("待命的費用是什麼")

    assert "135.2.e.5" in result.citations_markdown


def test_symbol_expansion_combines_definition_subtree():
    header = make_keyword_header_doc("811", "待命", "Hidden")
    sub_rule = make_rule_doc("811.1.b", "支付 [A]")
    symbol_def = make_rule_doc("135.2.e.5", "簡稱為 [A]。")
    symbol_sub = make_rule_doc("135.2.e.5.a", "當費用要求 [A] 時，可以任意屬性的力量支付。")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, rules=[header, sub_rule, symbol_def, symbol_sub])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=6, score_threshold=0.45)

    retrieved = chain._retrieve("待命的費用")

    matches = [doc for doc, _ in retrieved if doc.metadata.get("rule_id") == "135.2.e.5"]
    assert len(matches) == 1
    assert "當費用要求 [A] 時" in matches[0].page_content


def test_symbol_expansion_skips_already_included_definition():
    already_included = make_rule_doc("135.2.e.5", "簡稱為 [A]。支付 [A] 也在這裡。")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, rules=[already_included])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=6, score_threshold=0.45)

    expansions = chain._symbol_expansions([already_included])

    assert expansions == []


def test_no_symbol_in_matches_means_no_expansion():
    header = make_keyword_header_doc("805", "疾行", "Accelerate")
    sub_rule = make_rule_doc("805.1", "疾行是一種單位能力，不涉及任何符號。")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, rules=[header, sub_rule])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=6, score_threshold=0.45)

    retrieved = chain._retrieve("疾行是什麼")

    assert len(retrieved) == 1


def test_card_and_rule_indexes_are_loaded_once_at_construction():
    card = make_card_doc("OGN-245", "團結之印", "團結之印效果文字")
    header = make_keyword_header_doc("811", "待命", "Hidden")
    vectorstore = FakeVectorstore({"rule": [], "card": []}, cards=[card], rules=[header])
    chain = RiftboundRagChain(vectorstore=vectorstore, llm=FakeLLM("答案"), pool_per_type=10, k=6, score_threshold=0.45)

    assert vectorstore.bulk_fetch_calls == [{"source_type": "card"}, {"source_type": "rule"}]

    chain.ask("團結之印是什麼")
    chain.ask("另一個問題")

    assert vectorstore.bulk_fetch_calls == [{"source_type": "card"}, {"source_type": "rule"}]


def test_similarity_pool_keeps_slots_when_exact_matches_fill_k():
    # 4 card names + 4 keyword sections already exceed k=6; truncating the
    # whole merged list dropped every vector-ranked result.
    cards = [make_card_doc(f"C{i}", f"獨特卡牌{i}號", f"卡{i}效果") for i in range(4)]
    keywords = [make_keyword_header_doc(f"80{i}", f"關鍵字{i}", f"Keyword{i}") for i in range(4)]
    question = "".join(c.metadata["name_zh"] for c in cards) + "".join(
        k.page_content for k in keywords
    )
    pool_hit = make_rule_doc("999", "相似度命中的規則")
    vectorstore = FakeVectorstore(
        {"rule": [(pool_hit, 0.9)], "card": []}, cards=cards, rules=keywords
    )
    chain = RiftboundRagChain(
        vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=6, score_threshold=0.5
    )

    retrieved = chain._retrieve(question)

    assert pool_hit in [doc for doc, _ in retrieved]
    assert len([doc for doc, _ in retrieved if doc in cards]) == MAX_EXACT_MATCH_NAMES


def test_symbol_expansions_are_never_evicted_by_other_exact_matches():
    # The symbol expansion is what closes multi-hop questions, and it sat
    # last in the merged list where truncation reached it first.
    keyword = FakeDoc(
        page_content="待命（Hidden）\n支付 [A] 以使用。",
        metadata={"source_type": "rule", "rule_id": "811", "title": "待命（Hidden）"},
    )
    symbol_def = make_rule_doc("135", "力量簡稱為 [A]。")
    cards = [make_card_doc(f"C{i}", f"獨特卡牌{i}號", f"卡{i}效果") for i in range(4)]
    question = "待命（Hidden）" + "".join(c.metadata["name_zh"] for c in cards)
    vectorstore = FakeVectorstore(
        {"rule": [], "card": []}, cards=cards, rules=[keyword, symbol_def]
    )
    chain = RiftboundRagChain(
        vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=6, score_threshold=0.5
    )

    rule_ids = [doc.metadata.get("rule_id") for doc, _ in chain._retrieve(question)]

    assert "135" in rule_ids


def test_rule_subtrees_are_ordered_numerically_not_lexically():
    # Segment-wise string comparison put 805.10 ahead of 805.2.
    header = make_keyword_header_doc("805", "疾行", "Accelerate")
    rules = [header] + [make_rule_doc(f"805.{n}", f"第 {n} 條") for n in (10, 2)]
    vectorstore = FakeVectorstore({"rule": [], "card": []}, rules=rules)
    chain = RiftboundRagChain(
        vectorstore=vectorstore, llm=FakeLLM(""), pool_per_type=10, k=6, score_threshold=0.5
    )

    combined = chain.indexes.rule_docs_by_keyword["疾行"].page_content

    assert combined.index("第 2 條") < combined.index("第 10 條")


def test_a_card_document_without_an_id_does_not_break_printing_choice():
    # Every other metadata read uses .get(); this one raised at query time.
    doc = FakeDoc(page_content="效果", metadata={"source_type": "card", "name_zh": "無編號卡"})

    assert RiftboundRagChain._pick_printing([doc]) is doc


def test_citation_numbers_line_up_with_the_context_block():
    # The prompt tells the model to cite 「[1]」「[2]」 by context position, so
    # the source list has to carry those same numbers.
    rule = make_rule_doc("805.1", "疾行是一種單位能力。")
    card = make_card_doc("OGN-245", "團結之印", "團結之印效果")
    vectorstore = FakeVectorstore({"rule": [(rule, 0.9)], "card": [(card, 0.8)]})
    llm = FakeLLM("根據 [1] 與 [2]，答案是……")
    chain = RiftboundRagChain(
        vectorstore=vectorstore, llm=FakeLLM("x"), pool_per_type=10, k=6, score_threshold=0.5
    )
    chain.llm = llm

    result = chain.ask("疾行與團結之印")

    context = llm.last_messages[-1][1]
    assert "[1] 疾行是一種單位能力。" in context
    assert result.citations_markdown.startswith("[1] 規則 805.1")
    assert "[2] 卡牌《團結之印》" in result.citations_markdown


def test_rule_sort_key_orders_mixed_numeric_and_alphabetic_segments():
    ids = ["135.2.e.10", "135.2.e.2", "135.2.e.2.a", "135.10", "135.2"]

    assert sorted(ids, key=rule_sort_key) == [
        "135.2",
        "135.2.e.2",
        "135.2.e.2.a",
        "135.2.e.10",
        "135.10",
    ]
