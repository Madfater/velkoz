from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import structlog
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from riftbound_bot.rag.citations import citation_marker, format_citations
from riftbound_bot.rag.vectorstore import RetrievalStore

logger = structlog.get_logger("riftbound_bot.chain")

# A top-level (dot-free) rule chunk whose entire title is "CJK term（English
# gloss）" with no body of its own — the same shape as 805 疾行（Accelerate）,
# 809 偏斜（Deflect）, 811 待命（Hidden）. Used to detect rule sections that,
# like card names, have a short proper-noun-like term a user is likely to
# type verbatim.
_KEYWORD_TITLE_RE = re.compile(r"^(.+)（([A-Za-z][A-Za-z\s\-]*)）$")

# A bracket shorthand token as used throughout rules text, e.g. "[A]", "[C]".
_SYMBOL_RE = re.compile(r"\[([A-Za-z])\]")

# The corpus's own consistent phrasing for introducing a symbol's shorthand
# (134.2.a-f for domain symbols, 135.2.e.2/.3/.5/.6 for the rest) — used to
# find which rule defines a given "[X]" structurally, not via a hardcoded
# symbol table. Deliberately doesn't match "簡稱標示為 [T]"-style deprecated
# aliases (135.2.e.2/.3's parenthetical asides use a different phrase).
_SYMBOL_DEFINITION_RE = re.compile(r"簡稱為\s*\[([A-Za-z])\]")

# Sentinel score for force-included exact-name matches — deliberately not a
# real cosine value, so it can never accidentally tie/compare against a
# genuine similarity score from _search_pool.
EXACT_MATCH_SCORE = float("inf")

# Caps how many distinct card names a single question can force-include, so a
# pathological multi-name question can't crowd out every vector-ranked slot in
# k. Kept above 1-2 on purpose: the system prompt's own worked example
# ("我打出 X，對手有 Y") names two cards at once.
MAX_EXACT_MATCH_NAMES = 4

SYSTEM_PROMPT = """\
你是一個 Riftbound（符文之地）集換式卡牌遊戲的規則裁判助手，服務對象是繁體中文玩家。

規則：
1. 只能根據使用者訊息後方提供的「檢索內容」回答問題，不可以使用檢索內容以外的知識或臆測規則。
2. 回答中的每一個判斷或論點，都必須標註它所依據的來源編號，格式為「[來源1]」「[來源2]」等，
   對應到檢索內容列出的編號。
3. 如果檢索內容不足以回答問題（例如問題牽涉到的規則或卡牌沒有出現在檢索內容中），
   必須明確告訴使用者「目前收錄的資料無法回答這個問題」，並可以請使用者換個問法或補充
   更多細節，絕對不要憑空猜測或編造規則。
4. 若使用者描述了一個具體的牌局情境（例如「我打出 X，對手有 Y」），請根據檢索到的規則與
   卡牌文字，逐步解釋交互結果，並在每一步驟標註依據。
5. 一律使用繁體中文回答，維持簡潔、精確的裁判用語。
6. 檢索內容中的方括號符號（例如 [A]、[C]、[R]、[1]）是遊戲本身的費用與屬性符號，
   屬於規則原文的一部分，不是來源編號。來源編號一律寫成「[來源N]」。
7. 回答中第一次出現的遊戲符號或遊戲術語，都必須讓玩家看得懂：
   a. 若檢索內容有定義該符號或術語，就在後面用括號附上簡短說明，格式為
      「該符號（檢索內容中對它的定義）」；說明必須逐字取自檢索內容，不可自行改寫或補充。
      同一個符號或術語之後再次出現時不用重複說明。
   b. 若檢索內容只是提到該術語、但沒有定義它（例如「英雄區域」只出現在其他規則的引文中），
      就在後面標註「（檢索內容未收錄此術語的定義）」，不要自行補充或猜測它的意思。
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
    _SYMBOL_DEFINITION_RE), not a general multi-hop retrieval mechanism.
    """

    def __init__(
        self,
        vectorstore: RetrievalStore,
        llm: BaseChatModel,
        pool_per_type: int = 10,
        k: int = 6,
        score_threshold: float = 0.45,
    ) -> None:
        self.vectorstore = vectorstore
        self.llm = llm
        self.pool_per_type = pool_per_type
        self.k = k
        self.score_threshold = score_threshold
        self._card_docs_by_name = self._load_card_docs_by_name()
        rule_docs = self._load_rule_chunks()
        self._rule_docs_by_keyword = self._index_rule_subtrees(rule_docs, _KEYWORD_TITLE_RE, lambda m: m.group(1))
        self._rule_docs_by_symbol = self._index_rule_subtrees(
            rule_docs, _SYMBOL_DEFINITION_RE, lambda m: m.group(1), search=True
        )

    def _load_card_docs_by_name(self) -> dict[str, list[Document]]:
        """Loaded once from the indexed documents themselves (not the `cards`
        table) so this can never drift from what's actually searchable.

        fetch_by_source_type is a plain filtered read with no query vector.
        Under TurboVec this had to be faked with an empty query and a huge k,
        because that store had no bulk filtered-fetch API — which cost a
        throwaway embedding call per source type on every bot startup.
        """
        docs = self.vectorstore.fetch_by_source_type("card")
        by_name: dict[str, list[Document]] = {}
        for doc in docs:
            name = doc.metadata.get("name_zh")
            if not name:
                continue
            by_name.setdefault(name, []).append(doc)
        return by_name

    def _load_rule_chunks(self) -> list[Document]:
        return self.vectorstore.fetch_by_source_type("rule")

    @staticmethod
    def _combine_subtree(root_id: str, title: str, chunks: list[Document]) -> Document:
        """Combines a rule id and everything nested under it (rule_id ==
        root_id or starting with "root_id.") into one Document, sorted so
        sub-rules stay in their natural reading order. Individual sub-rules
        are often too short/generic to embed well on their own (see
        build_index.py's topic-context-prepending comment for the same
        problem one level up) — one combined document per matched section
        mirrors one document per card name in _load_card_docs_by_name.
        """
        subtree = sorted(
            (
                (doc.metadata.get("rule_id", ""), doc.page_content)
                for doc in chunks
                if doc.metadata.get("rule_id") == root_id
                or doc.metadata.get("rule_id", "").startswith(f"{root_id}.")
            ),
            key=lambda item: tuple(item[0].split(".")),
        )
        return Document(
            page_content="\n".join(content for _id, content in subtree),
            metadata={"source_type": "rule", "rule_id": root_id, "title": title},
        )

    def _index_rule_subtrees(
        self, chunks: list[Document], pattern: re.Pattern, extract_key, search: bool = False
    ) -> dict[str, Document]:
        """Shared engine behind _rule_docs_by_keyword and _rule_docs_by_symbol:
        find every chunk whose own text matches `pattern` (a keyword's
        top-level title, or a symbol-definition phrase), and combine that
        chunk's rule and its full subtree into one Document keyed by
        whatever `extract_key` pulls out of the match (the keyword term, or
        the symbol letter). `search=True` scans anywhere in a chunk's text
        (symbol definitions are usually mid-sentence); otherwise the whole
        title must match (keyword headers are exactly "term（gloss）", no
        more, no less — a `search` here would false-positive on inline
        parentheticals like rule 103.1's "一張英雄傳奇（Champion Legend）").
        """
        by_key: dict[str, Document] = {}
        for doc in chunks:
            title = doc.metadata.get("title", "")
            match = (pattern.search if search else pattern.match)(doc.page_content if search else title)
            if not match:
                continue
            root_id = doc.metadata.get("rule_id", "")
            if not search and "." in root_id:
                continue
            by_key[extract_key(match)] = self._combine_subtree(root_id, title, chunks)
        return by_key

    @staticmethod
    def _pick_printing(printings: list[Document]) -> Document:
        """A card name can map to multiple printings (reprints/alt-art share
        one name_zh). Prefer a non-異畫 (alt-art) printing; tie-break on the
        lowest card_id so the pick is deterministic rather than dependent on
        vectorstore return order.
        """
        non_alt = [doc for doc in printings if doc.metadata.get("rarity") != "異畫"]
        pool = non_alt or printings
        return min(pool, key=lambda doc: doc.metadata["card_id"])

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
        matches = self._specific_substring_matches(self._card_docs_by_name, question)
        return [
            (self._pick_printing(self._card_docs_by_name[name]), EXACT_MATCH_SCORE)
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
        matches = self._specific_substring_matches(self._rule_docs_by_keyword, question)
        return [(self._rule_docs_by_keyword[name], EXACT_MATCH_SCORE) for name in matches[:MAX_EXACT_MATCH_NAMES]]

    def _symbol_expansions(self, docs: list[Document]) -> list[tuple[Document, float]]:
        """When a retrieved document's own text uses a bracket shorthand
        (e.g. 811's cost text says "支付 [A]"), pull in whatever rule defines
        that symbol too, if it isn't already among `docs`. Confirmed live:
        811's force-included text mentions [A], but [A]'s own definition
        (135.2.e.5.a — the specific rule that would let the bot answer "yes"
        instead of "cannot determine") doesn't score well enough on
        similarity alone to make the pool on its own, even after topic-
        context prepending (see build_index.py). This is a one-hop expansion
        over a small, closed, well-defined symbol vocabulary — not a general
        multi-hop retrieval mechanism.

        `docs` is every document heading for the prompt, not just the
        force-included ones: the system prompt requires the answer to gloss
        each symbol it mentions on first use, and a symbol carried in by a
        similarity-ranked chunk needs its definition just as much as one
        carried in by an exact match. Without it the model is stuck between
        the grounding rule and the glossing rule, and correctly reports the
        symbol as undefined while its definition sits unretrieved in the
        index.
        """
        included_ids = {doc.metadata.get("rule_id") for doc in docs}
        symbols: set[str] = set()
        for doc in docs:
            symbols.update(_SYMBOL_RE.findall(doc.page_content))

        expansions = []
        for symbol in sorted(symbols):
            definition = self._rule_docs_by_symbol.get(symbol)
            if definition and definition.metadata.get("rule_id") not in included_ids:
                expansions.append((definition, EXACT_MATCH_SCORE))
                included_ids.add(definition.metadata.get("rule_id"))
        return expansions[:MAX_EXACT_MATCH_NAMES]

    def _search_pool(self, question: str, source_type: str) -> list[tuple[object, float]]:
        results = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=self.pool_per_type, filter={"source_type": source_type}
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

    @classmethod
    def _without_rule_subtrees(cls, pool: list[tuple], matched_top_ids: set[str]) -> list[tuple]:
        """Drops pool entries already covered by a force-included subtree.
        _combine_subtree folds a rule and everything under it into one
        document, so leaving the individual chunk in would repeat that text
        in the prompt and burn a second slot on it.
        """
        return [
            item
            for item in pool
            if not cls._in_matched_rule_subtree(item[0].metadata.get("rule_id"), matched_top_ids)
        ]

    def _retrieve(self, question: str) -> list[tuple[object, float]]:
        exact_cards = self._exact_name_matches(question)
        exact_keywords = self._exact_keyword_matches(question)

        exact_card_ids = {doc.metadata.get("card_id") for doc, _ in exact_cards}
        exact_rule_ids = {doc.metadata.get("rule_id") for doc, _ in exact_keywords}

        pool = self._search_pool(question, "rule") + self._search_pool(question, "card")
        pool = [item for item in pool if item[0].metadata.get("card_id") not in exact_card_ids]
        pool = self._without_rule_subtrees(pool, exact_rule_ids)
        pool.sort(key=lambda item: item[1], reverse=True)

        # Expand over the documents that would actually have reached the
        # prompt, pool included — a symbol only ever mentioned by a
        # similarity-ranked chunk still has to be glossable. Each expansion
        # then displaces the lowest-scored pool entry, exactly as expansions
        # triggered by an exact match already did.
        candidates = (exact_cards + exact_keywords + pool)[: self.k]
        symbol_expansions = self._symbol_expansions([doc for doc, _ in candidates])
        expansion_ids = {doc.metadata.get("rule_id") for doc, _ in symbol_expansions}
        pool = self._without_rule_subtrees(pool, expansion_ids)

        retrieved = (exact_cards + exact_keywords + symbol_expansions + pool)[: self.k]
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
            f"{citation_marker(i + 1)} {doc.page_content}"
            for i, (doc, _score) in enumerate(retrieved)
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
