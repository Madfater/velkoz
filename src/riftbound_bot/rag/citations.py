from __future__ import annotations

from dataclasses import dataclass


def citation_marker(index: int) -> str:
    """The token an answer cites source `index` by.

    Deliberately not a bare "[N]": the rules corpus writes generic energy
    costs as bracketed digits too (805.1.a's "額外支付 [1][C]"), so a bare
    marker is indistinguishable from a cost inside the very text it labels.

    The chain prefixes each context-block chunk with this, the answer cites
    it, and format_citations numbers the footer with it — one definition so
    the three can't drift apart.
    """
    return f"[來源{index}]"


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
    entries: dict[str, tuple[Citation, list[int]]] = {}
    for index, metadata in enumerate(metadatas, start=1):
        citation = citation_from_metadata(metadata)
        entries.setdefault(citation.label, (citation, []))[1].append(index)
    return "\n".join(
        "".join(citation_marker(index) for index in indices) + f" {citation.as_markdown()}"
        for citation, indices in entries.values()
    )
