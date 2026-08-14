"""Metadata keys shared between the index builder and everything reading it.

build_index writes these onto every Document; chain.py and citations.py read
them back. As bare string literals in four files, renaming one degraded
retrieval silently — no exception, just empty citations and filters that
stop matching — so producer and consumers name them from here instead.
"""
from __future__ import annotations

SOURCE_TYPE = "source_type"
RULE_ID = "rule_id"
TITLE = "title"
# Written onto every rule document and deliberately never read: it records
# which Markdown file a rule came from, which is worth having in the index
# for provenance when debugging the corpus.
SOURCE_FILE = "source_file"

CARD_ID = "card_id"
NAME_ZH = "name_zh"
NAME_EN = "name_en"
RARITY = "rarity"
SOURCE_URL = "source_url"

# Values of SOURCE_TYPE — the two corpora searched as separate pools.
RULE = "rule"
CARD = "card"

# The rarity chroniclecore and the API both use for alternate-art printings,
# which carry less text than the base printing (see _pick_printing).
ALT_ART_RARITY = "異畫"
