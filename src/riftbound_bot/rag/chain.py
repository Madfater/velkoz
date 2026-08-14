from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import structlog
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from riftbound_bot.rag import metadata as meta
from riftbound_bot.rag.citations import format_citations
from riftbound_bot.rag.lexical_index import LexicalIndexes

logger = structlog.get_logger("riftbound_bot.chain")

# A bracket shorthand token as used throughout rules text, e.g. "[A]", "[C]".
_SYMBOL_RE = re.compile(r"\[([A-Za-z])\]")

# Sentinel score for force-included exact-name matches — deliberately not a
# real cosine value, so it can never accidentally tie/compare against a
# genuine similarity score from _search_pool.
EXACT_MATCH_SCORE = float("inf")

# Caps how many distinct card names a single question can force-include, so a
# pathological multi-name question can't crowd out every vector-ranked slot in
# k. Kept above 1-2 on purpose: the system prompt's own worked example
# ("我打出 X，對手有 Y") names two cards at once.
MAX_EXACT_MATCH_NAMES = 4

# Slots the similarity pool keeps even when exact matches alone fill k.
# Exact matches are deterministic lexical facts and are never dropped, so
# without a floor a question naming several cards and keywords could evict
# the vector-ranked results entirely — including the symbol expansions that
# close multi-hop questions. k is therefore a target, not a hard cap: the
# worst case is 3 categories x MAX_EXACT_MATCH_NAMES plus this floor.
MIN_POOL_SLOTS = 2

SYSTEM_PROMPT = """\
你是一個 Riftbound（符文之地）集換式卡牌遊戲的規則裁判助手，服務對象是繁體中文玩家。

規則：
1. 只能根據使用者訊息後方提供的「檢索內容」回答問題，不可以使用檢索內容以外的知識或臆測規則。
2. 回答中的每一個判斷或論點，都必須標註它所依據的來源編號，格式為「[1]」「[2]」等，
   對應到檢索內容列出的編號。
3. 如果檢索內容不足以回答問題（例如問題牽涉到的規則或卡牌沒有出現在檢索內容中），
   必須明確告訴使用者「目前收錄的資料無法回答這個問題」，並可以請使用者換個問法或補充
   更多細節，絕對不要憑空猜測或編造規則。
4. 若使用者描述了一個具體的牌局情境（例如「我打出 X，對手有 Y」），請根據檢索到的規則與
   卡牌文字，逐步解釋交互結果，並在每一步驟標註依據。
5. 一律使用繁體中文回答，維持簡潔、精確的裁判用語。
"""

NO_CONTEXT_REPLY = (
    "目前收錄的規則與卡牌資料中，找不到與這個問題直接相關的內容，無法負責任地回答。"
    "可以試著換個問法，或補充更多細節（例如完整卡牌名稱、關鍵字）嗎？"
)


@dataclass(frozen=True)
class RagResult:
    answer: str
    citations_markdown: str
    metadatas: list[dict] = field(default_factory=list)


class RiftboundRagChain:
    """Retrieves rules and cards as two separate pools rather than one blended
    similarity search. The card corpus outnumbers the rules corpus ~25 to 1
    (1,256 vs ~51 chunks) and is structurally homogeneous (short, similarly-
    shaped strings) — verified empirically that a single combined top-k
    search lets cards drown out genuinely more relevant rule chunks, even
    though each corpus individually ranks cleanly (the true match for a
    keyword query ranked #1 within the rules-only pool, with a clean score
    gap over unrelated content). Searching each source_type independently
    and merging by score fixes that without needing a reranking step.

    Separately, even within the card-only pool, embedding similarity alone
    has been confirmed unreliable for exact-card-name queries — a card whose
    full name is typed verbatim in the question can still rank far outside
    the top-k similarity results despite its name being in its own indexed
    text. _exact_name_matches works around that with a deterministic,
    zero-network substring check that force-includes such cards ahead of
    the similarity-ranked pool.

    The same problem was independently confirmed for rule keyword sections
    (e.g. 待命/Hidden) on multi-hop interaction questions — even with
    topic-context prepending (see build_index.py), a keyword's decisive
    sub-rule can still rank outside the pool. _exact_keyword_matches applies
    the identical substring-match workaround to rule sections shaped like
    card names (a short "CJK term（English gloss）" top-level title).

    Even that isn't always enough: a keyword's own cost text can reference a
    bracket symbol (e.g. 811's "支付 [A]") whose *own* definition
    (135.2.e.5.a) still doesn't score well enough to surface on similarity
    alone — confirmed live even after generalizing build_index.py's
    topic-context prepending to the full ancestor-heading chain.
    _symbol_expansions handles this last mile: a one-hop expansion over the
    small, closed set of bracket symbols this corpus defines (see
    lexical_index.SYMBOL_DEFINITION_RE), not a general multi-hop retrieval
    mechanism.

    The three lexical lookups themselves are built in rag/lexical_index.py.
    README's "Known issue: card retrieval quality" section records the
    measurements behind all of this, including what remains unsolved for
    fuzzy and attribute-only queries.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseChatModel,
        pool_per_type: int = 10,
        k: int = 6,
        score_threshold: float = 0.5,
    ) -> None:
        self.vectorstore = vectorstore
        self.llm = llm
        self.pool_per_type = pool_per_type
        self.k = k
        self.score_threshold = score_threshold
        self.indexes = LexicalIndexes(vectorstore)

    @staticmethod
    def _pick_printing(printings: list[Document]) -> Document:
        """A card name can map to multiple printings (reprints/alt-art share
        one name_zh). Prefer a non-異畫 (alt-art) printing; tie-break on the
        lowest card_id so the pick is deterministic rather than dependent on
        vectorstore return order.
        """
        non_alt = [doc for doc in printings if doc.metadata.get(meta.RARITY) != meta.ALT_ART_RARITY]
        pool = non_alt or printings
        return min(pool, key=lambda doc: doc.metadata.get(meta.CARD_ID, ""))

    @staticmethod
    def _specific_substring_matches(candidates, question: str) -> list[str]:
        """Longest-candidate-first substring scan with subsumption filtering:
        if both "提摩" and "提摩-戰略家" match the same question, only the
        more specific "提摩-戰略家" is kept — real, confirmed collision
        shapes in the card corpus, not a hypothetical edge case.
        """
        names = sorted(candidates, key=len, reverse=True)
        raw_matches = [name for name in names if name in question]
        return [
            name
            for name in raw_matches
            if not any(name != other and name in other for other in raw_matches)
        ]

    def _exact_name_matches(self, question: str) -> list[tuple[Document, float]]:
        """Force-includes a card's document when its full name_zh appears
        literally in the question, regardless of similarity score. Exists
        because an exact-name query can still rank a card far outside the
        normal top-k similarity pool (see class docstring) even though the
        name is present verbatim in its own indexed text — a deterministic
        lexical fact that a similarity score shouldn't be trusted to surface
        reliably on its own.
        """
        matches = self._specific_substring_matches(self.indexes.card_docs_by_name, question)
        return [
            (self._pick_printing(self.indexes.card_docs_by_name[name]), EXACT_MATCH_SCORE)
            for name in matches[:MAX_EXACT_MATCH_NAMES]
        ]

    def _exact_keyword_matches(self, question: str) -> list[tuple[Document, float]]:
        """Same mechanism as _exact_name_matches, for rule keyword sections
        instead of card names — confirmed live that keyword sub-rules (e.g.
        811.1.b, the specific rule that answers a cost-payment question) can
        rank far outside the similarity pool even with topic-context
        prepending, when the question is a multi-hop interaction question
        rather than a direct "what is X" definition question.
        """
        matches = self._specific_substring_matches(self.indexes.rule_docs_by_keyword, question)
        return [
            (self.indexes.rule_docs_by_keyword[name], EXACT_MATCH_SCORE)
            for name in matches[:MAX_EXACT_MATCH_NAMES]
        ]

    def _symbol_expansions(self, docs: list[Document]) -> list[tuple[Document, float]]:
        """When a force-included document's own text uses a bracket shorthand
        (e.g. 811's cost text says "支付 [A]"), pull in whatever rule defines
        that symbol too, if it isn't already among `docs`. Confirmed live:
        811's force-included text mentions [A], but [A]'s own definition
        (135.2.e.5.a — the specific rule that would let the bot answer "yes"
        instead of "cannot determine") doesn't score well enough on
        similarity alone to make the pool on its own, even after topic-
        context prepending (see build_index.py). This is a one-hop expansion
        over a small, closed, well-defined symbol vocabulary — not a general
        multi-hop retrieval mechanism.
        """
        included_ids = {doc.metadata.get(meta.RULE_ID) for doc in docs}
        symbols: set[str] = set()
        for doc in docs:
            symbols.update(_SYMBOL_RE.findall(doc.page_content))

        expansions = []
        for symbol in sorted(symbols):
            definition = self.indexes.rule_docs_by_symbol.get(symbol)
            if definition and definition.metadata.get(meta.RULE_ID) not in included_ids:
                expansions.append((definition, EXACT_MATCH_SCORE))
                included_ids.add(definition.metadata.get(meta.RULE_ID))
        return expansions[:MAX_EXACT_MATCH_NAMES]

    def _search_pool(self, question: str, source_type: str) -> list[tuple[Document, float]]:
        results = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=self.pool_per_type, filter={meta.SOURCE_TYPE: source_type}
        )
        scored = [(doc, score) for doc, score in results if score >= self.score_threshold]
        logger.debug(
            "chain.search_pool",
            source_type=source_type,
            candidates=len(results),
            kept=len(scored),
            score_min=min((s for _d, s in results), default=None),
            score_max=max((s for _d, s in results), default=None),
        )
        return scored

    @staticmethod
    def _in_matched_rule_subtree(rule_id: str | None, matched_top_ids: set[str]) -> bool:
        if not rule_id:
            return False
        return rule_id in matched_top_ids or any(rule_id.startswith(f"{top}.") for top in matched_top_ids)

    def _retrieve(self, question: str) -> list[tuple[Document, float]]:
        exact_cards = self._exact_name_matches(question)
        exact_keywords = self._exact_keyword_matches(question)
        symbol_expansions = self._symbol_expansions([doc for doc, _ in exact_cards + exact_keywords])

        exact_card_ids = {doc.metadata.get(meta.CARD_ID) for doc, _ in exact_cards}
        exact_rule_ids = {doc.metadata.get(meta.RULE_ID) for doc, _ in exact_keywords + symbol_expansions}

        pool = self._search_pool(question, meta.RULE) + self._search_pool(question, meta.CARD)
        pool = [
            item
            for item in pool
            if item[0].metadata.get(meta.CARD_ID) not in exact_card_ids
            and not self._in_matched_rule_subtree(item[0].metadata.get(meta.RULE_ID), exact_rule_ids)
        ]
        pool.sort(key=lambda item: item[1], reverse=True)

        # Truncate the pool rather than the concatenation: slicing the whole
        # list to k let 3 categories of up to MAX_EXACT_MATCH_NAMES each crowd
        # out everything behind them, so a question naming several cards could
        # drop the symbol expansions and the similarity results entirely.
        exact = exact_cards + exact_keywords + symbol_expansions
        pool_slots = max(self.k - len(exact), MIN_POOL_SLOTS)
        retrieved = exact + pool[:pool_slots]
        logger.info(
            "chain.retrieve",
            exact_card_matches=len(exact_cards),
            exact_keyword_matches=len(exact_keywords),
            symbol_expansions=len(symbol_expansions),
            pool_candidates=len(pool),
            retrieved=len(retrieved),
        )
        return retrieved

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> RagResult:
        retrieve_start = time.monotonic()
        retrieved = self._retrieve(question)
        retrieve_seconds = time.monotonic() - retrieve_start
        if not retrieved:
            logger.info("chain.ask.no_context", retrieve_seconds=round(retrieve_seconds, 3))
            return RagResult(answer=NO_CONTEXT_REPLY, citations_markdown="")

        context_block = "\n\n".join(
            f"[{i + 1}] {doc.page_content}" for i, (doc, _score) in enumerate(retrieved)
        )
        messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT)]
        messages.extend(history or [])
        messages.append(
            (
                "human",
                f"檢索內容：\n\n{context_block}\n\n使用者問題：{question}",
            )
        )

        generate_start = time.monotonic()
        response = self.llm.invoke(messages)
        generate_seconds = time.monotonic() - generate_start
        logger.info(
            "chain.ask.done",
            retrieved=len(retrieved),
            retrieve_seconds=round(retrieve_seconds, 3),
            generate_seconds=round(generate_seconds, 3),
        )
        metadatas = [doc.metadata for doc, _score in retrieved]
        return RagResult(
            answer=str(response.content),
            citations_markdown=format_citations(metadatas),
            metadatas=metadatas,
        )
