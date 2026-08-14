"""CLI: (re)builds the TurboVec vector store from the `rules` and `cards`
Postgres tables (populated by rules_sync.py / cards_scrape.py /
cards_from_api.py). Manual, re-runnable refresh — no ingestion pipeline or
watch process, matching the design doc's "fixed snapshot" data model.

Usage:
    python -m riftbound_bot.ingest.build_index
"""
from __future__ import annotations

from pathlib import Path

import structlog
from langchain_core.documents import Document

from riftbound_bot.config import load_ingest_settings
from riftbound_bot.ingest.db import CARDS_TABLE, RULES_TABLE, get_connection
from riftbound_bot.ingest.rules_parser import RuleChunk
from riftbound_bot.logging_config import configure_logging
from riftbound_bot.rag import metadata as meta
from riftbound_bot.rag.vectorstore import build_embeddings, create_vectorstore

logger = structlog.get_logger("riftbound_bot.build_index")


def _rule_documents(conn) -> list[Document]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT data FROM {RULES_TABLE}")
        rows = [row[0] for row in cur.fetchall()]
    if not rows:
        logger.warning("build_index.no_rules", hint="run rules_sync.py first")
        return []
    return _rule_documents_from_chunks([RuleChunk(**row) for row in rows])


def _rule_documents_from_chunks(chunks: list[RuleChunk]) -> list[Document]:
    """Builds one Document per rule chunk, with its ancestor *headings*
    prepended to the embedded content (outermost first).

    Deeply-nested sub-rules are often just a few words on their own (e.g.
    805.1 is literally "疾行是一種單位（Unit）能力。") and embed almost
    meaninglessly without their topic attached — verified against the real
    index: 805.1 ranked 1306th out of 1307 documents for a query asking
    exactly what Accelerate/疾行 is, because "Accelerate" never appears in
    its own text. Prepending the section's own name fixes that.

    Only *pure heading* ancestors are prepended — short label-style titles
    like "805 疾行（Accelerate）" or "135.2.e 符號" — not every ancestor's
    full text. Walking the whole ancestor chain (not just the top-level one)
    matters once a section has real internal structure: 135.2.e.5.a sits
    four levels under the broad top-level "135 規則文字（Rules Text）", which
    is too generic to help; its immediate parent "135.2.e 符號" is the
    specific, useful context — confirmed live that 135.2.e.5.a ranked
    69th/94 for a question about paying an [A] cost with only the top-level
    heading attached.

    Every rule in this corpus is written as `[id] text` on one line, so
    RuleChunk.body is always empty and RuleChunk.title holds the full rule
    text regardless of whether that text is a short label or a full
    sentence — an "is there a body" check can't tell headings apart from
    rule statements. What does: genuine rule statements are grammatically
    complete sentences and end in 。／！／？; heading-style titles don't
    (they end mid-phrase, or on a closing "）" from an English gloss).
    Skipping non-heading ancestors avoids dragging a parent's full
    sentence — sometimes containing lengthy worked examples — into every
    descendant's embedding; this is also why 811's already-shallow subtree
    embeds identically to before (811 itself is the only heading-shaped
    ancestor any of its sub-rules have).
    """
    by_id = {chunk.rule_id: chunk for chunk in chunks}
    sentence_end = ("。", "！", "？")

    def topic_context(rule_id: str) -> str:
        parts = rule_id.split(".")
        ancestor_ids = (".".join(parts[:i]) for i in range(1, len(parts)))
        headings = [
            ancestor.title
            for ancestor_id in ancestor_ids
            if (ancestor := by_id.get(ancestor_id))
            and ancestor.title
            and not ancestor.body
            and not ancestor.title.rstrip().endswith(sentence_end)
        ]
        return "\n".join(headings)

    documents = []
    for chunk in chunks:
        if not chunk.text:
            continue
        context = topic_context(chunk.rule_id)
        embed_text = f"{context}\n{chunk.text}" if context else chunk.text
        documents.append(
            Document(
                page_content=embed_text,
                metadata={
                    meta.SOURCE_TYPE: meta.RULE,
                    meta.RULE_ID: chunk.rule_id,
                    meta.TITLE: chunk.title,
                    meta.SOURCE_FILE: chunk.source_file,
                },
            )
        )
    return documents


def _card_embed_text(card: dict) -> str:
    """The text a card is retrieved by.

    Includes colour, rarity, stats and trait tags rather than just
    name/id/category/effect: those attributes are the whole substance of
    questions like "哪些卡可以獲得黃色力量" or "紅色 3 費單位", and leaving
    them out of the embedded string made such questions unanswerable no
    matter how the similarity search was tuned.
    """
    stat_bits = []
    for label, key in (("費用", "energy"), ("力量", "power"), ("戰力", "might")):
        value = card.get(key)
        if value is not None:
            stat_bits.append(f"{label} {value}")
    stats = "、".join(stat_bits)
    tags = card.get("tags") or []
    tag_str = f"｜標籤：{'、'.join(tags)}" if tags else ""

    header = (
        f"{card.get('name_zh', '')}（{card.get('name_en', '')}）"
        f"｜{card.get('id', '')}｜{card.get('category', '')}"
        f"｜顏色：{card.get('color', '')}｜稀有度：{card.get('rarity', '')}"
        f"{'｜' + stats if stats else ''}{tag_str}"
    )
    if card.get("rules_text_zh"):
        return f"{header}\n效果：{card['rules_text_zh']}"
    return header


def _card_documents(conn) -> list[Document]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT data FROM {CARDS_TABLE}")
        cards = [row[0] for row in cur.fetchall()]
    if not cards:
        logger.warning("build_index.no_cards", hint="run cards_scrape.py first")
        return []
    return _card_documents_from_dicts(cards)


def _card_documents_from_dicts(cards: list[dict]) -> list[Document]:
    documents = []
    for card in cards:
        # A row missing its id or name can't be cited or looked up by name, so
        # it's worth skipping loudly rather than aborting a whole rebuild on
        # one malformed record.
        if not card.get("id") or not card.get("name_zh"):
            logger.warning("build_index.card_skipped", card_id=card.get("id"))
            continue
        documents.append(
            Document(
                page_content=_card_embed_text(card),
                metadata={
                    meta.SOURCE_TYPE: meta.CARD,
                    meta.CARD_ID: card["id"],
                    meta.NAME_ZH: card["name_zh"],
                    meta.NAME_EN: card.get("name_en", ""),
                    meta.RARITY: card.get("rarity", ""),
                    meta.SOURCE_URL: card.get("source_url", ""),
                },
            )
        )
    return documents


def build_index(settings) -> int:
    with get_connection(settings) as conn:
        documents = _rule_documents(conn) + _card_documents(conn)
    if not documents:
        raise RuntimeError(
            "No documents found to index — run rules_sync.py / cards_scrape.py first."
        )

    embeddings = build_embeddings(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
    )
    vectorstore = create_vectorstore(embeddings)
    # Embed in batches so progress is visible on a full rebuild. Nothing
    # touches disk until the final dump() below — a crash mid-loop leaves
    # whatever was already persisted at persist_dir completely untouched,
    # unlike Chroma's old auto-persist-per-write behavior which could leave
    # the bot silently serving a partial index after a crash.
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        vectorstore.add_documents(batch)
        logger.info("build_index.progress", indexed=min(start + batch_size, len(documents)), total=len(documents))

    persist_dir = Path(settings.vector_store_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.dump(str(persist_dir))

    return len(documents)


def main() -> None:
    configure_logging()
    settings = load_ingest_settings()
    count = build_index(settings)
    logger.info("build_index.done", indexed=count, persist_dir=settings.vector_store_dir)


if __name__ == "__main__":
    main()
