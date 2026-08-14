"""Lexical lookups built once from the vector store at startup.

These are the deterministic, zero-network half of retrieval: exact card
names, keyword rule sections, and bracket-symbol definitions, each mapping a
literal string a user might type to the document that answers it. The chain
consults them before trusting similarity scores — see RiftboundRagChain for
why each one exists.

Built from the vector store rather than Postgres so they can never describe
documents that aren't actually indexed and searchable.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from riftbound_bot.rag import metadata as meta

# A top-level (dot-free) rule chunk whose entire title is "CJK term（English
# gloss）" with no body of its own — the same shape as 805 疾行（Accelerate）,
# 809 偏斜（Deflect）, 811 待命（Hidden）. Used to detect rule sections that,
# like card names, have a short proper-noun-like term a user is likely to
# type verbatim.
KEYWORD_TITLE_RE = re.compile(r"^(.+)（([A-Za-z][A-Za-z\s\-]*)）$")

# The corpus's own consistent phrasing for introducing a symbol's shorthand
# (134.2.a-f for domain symbols, 135.2.e.2/.3/.5/.6 for the rest) — used to
# find which rule defines a given "[X]" structurally, not via a hardcoded
# symbol table. Deliberately doesn't match "簡稱標示為 [T]"-style deprecated
# aliases (135.2.e.2/.3's parenthetical asides use a different phrase).
SYMBOL_DEFINITION_RE = re.compile(r"簡稱為\s*\[([A-Za-z])\]")

# Upper bound for the "give me everything matching this filter" bulk-fetch.
# TurboQuantVectorStore's similarity_search returns the complete filtered set
# once k >= the candidate count — this just needs comfortable headroom over
# the corpus (~1,300 documents today), not an exact count.
BULK_FETCH_LIMIT = 20_000


def rule_sort_key(rule_id: str) -> tuple:
    """Orders dotted rule ids the way the rulebook does.

    Segment-wise string comparison put 805.10 before 805.2. Segments are
    mixed (135.2.e.5.a), so numeric ones sort as numbers and ahead of
    alphabetic ones at the same depth.
    """
    return tuple(
        (0, int(segment), "") if segment.isdigit() else (1, 0, segment)
        for segment in rule_id.split(".")
    )


def combine_subtree(root_id: str, title: str, chunks: list[Document]) -> Document:
    """Combines a rule id and everything nested under it (rule_id == root_id
    or starting with "root_id.") into one Document, sorted so sub-rules stay
    in their natural reading order. Individual sub-rules are often too
    short/generic to embed well on their own (see build_index.py's
    topic-context-prepending comment for the same problem one level up) — one
    combined document per matched section mirrors one document per card name
    in LexicalIndexes.card_docs_by_name.
    """
    subtree = sorted(
        (
            (doc.metadata.get(meta.RULE_ID, ""), doc.page_content)
            for doc in chunks
            if doc.metadata.get(meta.RULE_ID) == root_id
            or doc.metadata.get(meta.RULE_ID, "").startswith(f"{root_id}.")
        ),
        key=lambda item: rule_sort_key(item[0]),
    )
    return Document(
        page_content="\n".join(content for _id, content in subtree),
        metadata={meta.SOURCE_TYPE: meta.RULE, meta.RULE_ID: root_id, meta.TITLE: title},
    )


class LexicalIndexes:
    """Card names, keyword sections, and symbol definitions, keyed by the
    literal text a question would contain.

    Loaded once at construction. Costs two throwaway embedding calls at bot
    startup, not per request, and freezes with the index — a build_index
    rerun needs a bot restart to be seen.
    """

    def __init__(self, vectorstore: VectorStore) -> None:
        self.card_docs_by_name = self._load_card_docs_by_name(vectorstore)
        rule_docs = self._bulk_fetch(vectorstore, meta.RULE)
        self.rule_docs_by_keyword = self._index_rule_subtrees(
            rule_docs, KEYWORD_TITLE_RE, lambda m: m.group(1)
        )
        self.rule_docs_by_symbol = self._index_rule_subtrees(
            rule_docs, SYMBOL_DEFINITION_RE, lambda m: m.group(1), search=True
        )

    @staticmethod
    def _bulk_fetch(vectorstore: VectorStore, source_type: str) -> list[Document]:
        """Every indexed document of one source_type.

        Uses similarity_search with a throwaway query and a generous k rather
        than a bulk-fetch API: TurboQuantVectorStore has none, but its
        similarity_search returns the *complete* filtered set once k meets or
        exceeds the candidate count (confirmed against the installed package:
        the store builds an allow-list from every matching document first,
        then returns min(k, n_allowed) results), so a large BULK_FETCH_LIMIT
        gets everything and the unranked order doesn't matter here.
        """
        return vectorstore.similarity_search(
            "", k=BULK_FETCH_LIMIT, filter={meta.SOURCE_TYPE: source_type}
        )

    def _load_card_docs_by_name(self, vectorstore: VectorStore) -> dict[str, list[Document]]:
        """Indexed by name_zh, which is what a question spells out. One name
        can map to several printings — see the chain's _pick_printing."""
        by_name: dict[str, list[Document]] = {}
        for doc in self._bulk_fetch(vectorstore, meta.CARD):
            name = doc.metadata.get(meta.NAME_ZH)
            if not name:
                continue
            by_name.setdefault(name, []).append(doc)
        return by_name

    @staticmethod
    def _index_rule_subtrees(
        chunks: list[Document],
        pattern: re.Pattern,
        extract_key: Callable[[re.Match], str],
        search: bool = False,
    ) -> dict[str, Document]:
        """Shared engine behind rule_docs_by_keyword and rule_docs_by_symbol:
        find every chunk matching `pattern`, and combine that chunk's rule and
        its full subtree into one Document keyed by whatever `extract_key`
        pulls out of the match (the keyword term, or the symbol letter).

        `search=True` scans anywhere in a chunk's *text*, since symbol
        definitions are usually mid-sentence. Otherwise the whole *title* must
        match: keyword headers are exactly "term（gloss）", no more, no less,
        and a search here would false-positive on inline parentheticals like
        rule 103.1's "一張冠軍傳奇（Champion Legend）".
        """
        by_key: dict[str, Document] = {}
        for doc in chunks:
            title = doc.metadata.get(meta.TITLE, "")
            subject = doc.page_content if search else title
            match = pattern.search(subject) if search else pattern.match(subject)
            if not match:
                continue
            root_id = doc.metadata.get(meta.RULE_ID, "")
            if not search and "." in root_id:
                continue
            by_key[extract_key(match)] = combine_subtree(root_id, title, chunks)
        return by_key
