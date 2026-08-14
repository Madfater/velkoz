from __future__ import annotations

from dataclasses import dataclass

# Longest title still treated as a heading rather than inline rule prose. Rule
# text authored after the `[id]` marker lands in `title` too (see
# rules_parser), and a full sentence makes a poor citation label.
MAX_HEADING_TITLE_LENGTH = 24


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
        short_title = title if len(title) <= MAX_HEADING_TITLE_LENGTH else ""
        label = f"規則 {rule_id}" + (f" {short_title}" if short_title else "")
        return Citation(label=label)
    if source_type == "card":
        name_zh = metadata.get("name_zh", "")
        card_id = metadata.get("card_id", "")
        label = f"卡牌《{name_zh}》（{card_id}）"
        return Citation(label=label, source_url=metadata.get("source_url") or None)
    return Citation(label=metadata.get("title") or metadata.get("card_id") or "未知來源")


def format_citations(metadatas: list[dict]) -> str:
    """Numbered, de-duplicated citation list for the reply's embed field.

    The numbers are the positions of the retrieved documents in the context
    block the chain builds, which is what the system prompt tells the model
    to cite as 「[1]」「[2]」. Rendering these as a plain bullet list meant an
    answer saying 「根據 [3]」 pointed at nothing the reader could see.

    A label repeated across context slots keeps its first number, so the list
    can skip one (`[1]`, `[3]`) — the gap is honest: slot 2 cited the same
    source as slot 1.
    """
    numbered: dict[str, int] = {}
    lines: list[str] = []
    for index, metadata in enumerate(metadatas, start=1):
        citation = citation_from_metadata(metadata)
        if citation.label in numbered:
            continue
        numbered[citation.label] = index
        lines.append(f"[{index}] {citation.as_markdown()}")
    return "\n".join(lines)
