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

from riftbound_bot.config import Settings
from riftbound_bot.ingest.db import CARDS_TABLE, RULES_TABLE, get_connection
from riftbound_bot.ingest.rules_parser import RuleChunk
from riftbound_bot.logging_config import configure_logging
from riftbound_bot.rag.vectorstore import build_embeddings, create_vectorstore

logger = structlog.get_logger("riftbound_bot.build_index")


def _rule_documents(conn) -> list[Document]:
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
    with conn.cursor() as cur:
        cur.execute(f"SELECT data FROM {RULES_TABLE}")
        chunks = [RuleChunk(**row[0]) for row in cur.fetchall()]
    if not chunks:
        print("No rule data in Postgres — skipping rules (run rules_sync.py first).")
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
                    "source_type": "rule",
                    "rule_id": chunk.rule_id,
                    "title": chunk.title,
                    "source_file": chunk.source_file,
                },
            )
        )
    return documents


def _card_documents(conn) -> list[Document]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT data FROM {CARDS_TABLE}")
        cards = [row[0] for row in cur.fetchall()]
    if not cards:
        print("No card data in Postgres — skipping cards (run cards_scrape.py first).")
        return []
    documents = []
    for card in cards:
        text_parts = [
            f"{card['name_zh']}（{card['name_en']}）｜{card['id']}｜{card['category']}",
        ]
        if card.get("rules_text_zh"):
            text_parts.append(f"效果：{card['rules_text_zh']}")
        documents.append(
            Document(
                page_content="\n".join(text_parts),
                metadata={
                    "source_type": "card",
                    "card_id": card["id"],
                    "name_zh": card["name_zh"],
                    "name_en": card["name_en"],
                    "rarity": card.get("rarity", ""),
                    "source_url": card.get("source_url", ""),
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
    settings = Settings.load_for_ingest()
    count = build_index(settings)
    print(f"Done. Indexed {count} chunks into {settings.vector_store_dir}.")


if __name__ == "__main__":
    main()
