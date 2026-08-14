"""CLI: (re)builds the local Chroma vector store from data/rules/ and
data/cards/cards.json. Manual, re-runnable refresh — no ingestion pipeline
or watch process, matching the design doc's "fixed snapshot" data model.

Usage:
    python -m riftbound_bot.ingest.build_index
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_core.documents import Document

from riftbound_bot.config import Settings
from riftbound_bot.ingest.rules_parser import parse_rules_dir
from riftbound_bot.rag.vectorstore import get_vectorstore


def _rule_documents(rules_dir: str) -> list[Document]:
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
    chunks = parse_rules_dir(rules_dir)
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


def _card_documents(cards_file: str) -> list[Document]:
    path = Path(cards_file)
    if not path.exists():
        print(f"No card data at {cards_file} — skipping cards (run cards_scrape.py first).")
        return []
    cards = json.loads(path.read_text(encoding="utf-8"))
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
    persist_dir = Path(settings.chroma_persist_dir)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = _rule_documents(settings.rules_dir) + _card_documents(settings.cards_file)
    if not documents:
        raise RuntimeError("No documents found to index — check RULES_DIR / CARDS_FILE.")

    vectorstore = get_vectorstore(
        persist_dir=str(persist_dir),
        embedding_base_url=settings.embedding_base_url,
        embedding_api_key=settings.embedding_api_key,
        embedding_model=settings.embedding_model,
    )
    # Embed in batches so progress is visible on a full rebuild.
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        vectorstore.add_documents(batch)
        print(f"Indexed {min(start + batch_size, len(documents))}/{len(documents)}")

    return len(documents)


def main() -> None:
    settings = Settings.load_for_ingest()
    count = build_index(settings)
    print(f"Done. Indexed {count} chunks into {settings.chroma_persist_dir}.")


if __name__ == "__main__":
    main()
