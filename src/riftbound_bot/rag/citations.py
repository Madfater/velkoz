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
    """De-duplicated, ordered citation list for the reply's footer/embed field."""
    seen: set[str] = set()
    lines: list[str] = []
    for metadata in metadatas:
        citation = citation_from_metadata(metadata)
        if citation.label in seen:
            continue
        seen.add(citation.label)
        lines.append(f"- {citation.as_markdown()}")
    return "\n".join(lines)
