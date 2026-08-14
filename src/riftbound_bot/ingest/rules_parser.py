"""Parses the `[rule.id] title / body` Markdown convention documented at the
top of data/rules/core_rules_zh_tw.md into one chunk per rule, so every chunk
maps to exactly one citable rule number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_RULE_HEADER_RE = re.compile(r"^\[(?P<id>\d+(?:\.[A-Za-z0-9]+)*)\]\s*(?P<title>.*)$")


@dataclass(frozen=True)
class RuleChunk:
    rule_id: str
    title: str
    body: str
    source_file: str

    @property
    def text(self) -> str:
        """Text to embed: title folded in so section headers are searchable too."""
        return f"{self.title}\n{self.body}".strip() if self.title else self.body


def parse_rules_text(text: str, source_file: str = "") -> list[RuleChunk]:
    text = _COMMENT_RE.sub("", text)
    lines = text.splitlines()

    chunks: list[RuleChunk] = []
    current_id: str | None = None
    current_title = ""
    body_lines: list[str] = []

    def flush() -> None:
        if current_id is None:
            return
        body = "\n".join(body_lines).strip()
        chunks.append(
            RuleChunk(
                rule_id=current_id,
                title=current_title.strip(),
                body=body,
                source_file=source_file,
            )
        )

    for line in lines:
        match = _RULE_HEADER_RE.match(line.strip())
        if match:
            flush()
            current_id = match.group("id")
            current_title = match.group("title")
            body_lines = []
        elif current_id is not None:
            if line.strip():
                body_lines.append(line.strip())

    flush()
    return chunks


def parse_rules_dir(rules_dir: str | Path) -> list[RuleChunk]:
    rules_dir = Path(rules_dir)
    chunks: list[RuleChunk] = []
    for path in sorted(rules_dir.glob("*.md")):
        chunks.extend(parse_rules_text(path.read_text(encoding="utf-8"), source_file=path.name))
    return chunks
