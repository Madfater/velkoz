from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    label: str
    source_url: str | None = None

    def as_markdown(self) -> str:
        if self.source_url:
            return f"[{self.label}]({self.source_url})"
        return self.label


def citation_from_metadata(metadata: dict) -> Citation:
    source_type = metadata.get("source_type")
    if source_type == "rule":
        rule_id = metadata.get("rule_id", "")
        title = metadata.get("title", "")
        # Only surface short, heading-like titles in the citation label — long
        # ones mean the rule's full body text was authored inline after the
        # `[id]` marker rather than on a following line, and isn't a heading.
        short_title = title if len(title) <= 24 else ""
        label = f"規則 {rule_id}" + (f" {short_title}" if short_title else "")
        return Citation(label=label)
    if source_type == "card":
        name_zh = metadata.get("name_zh", "")
        card_id = metadata.get("card_id", "")
        label = f"卡牌《{name_zh}》（{card_id}）"
        return Citation(label=label, source_url=metadata.get("source_url") or None)
    return Citation(label=metadata.get("title") or metadata.get("card_id") or "未知來源")


def format_citations(metadatas: list[dict]) -> str:
    """De-duplicated citation list for the reply's footer/embed field, each
    entry carrying the "[來源N]" markers the answer text cites it by.

    The numbers are the 1-based positions in `metadatas`, which is aligned
    with the "[來源N]" prefixes the chain puts on the context block — not the
    positions of the rendered lines. Two retrieved chunks can share a label
    (the same rule reached twice), and numbering the output lines instead
    would silently shift every entry below such a collapse onto a number
    belonging to a different source.

    For the same reason a repeated label keeps *all* of its markers
    ("[來源2][來源5] 規則 805.1") rather than only the first: the model saw
    both slots in its context and may cite either one.
    """
    markers_by_label: dict[str, list[int]] = {}
    citations: dict[str, Citation] = {}
    for index, metadata in enumerate(metadatas, start=1):
        citation = citation_from_metadata(metadata)
        markers_by_label.setdefault(citation.label, []).append(index)
        citations.setdefault(citation.label, citation)
    return "\n".join(
        "".join(f"[來源{index}]" for index in markers) + f" {citations[label].as_markdown()}"
        for label, markers in markers_by_label.items()
    )
