"""Read-only corpora shipped inside the package, so a fresh deployment can
initialize itself with no local data directory and no third-party site.

These are *seeds*, not the source of truth. Postgres is the source of truth:
bootstrap loads a seed only into an empty table and never overwrites what's
already stored, so editing rules in the database survives every restart.

They live under src/ rather than a data/ folder on purpose. A bind-mounted
host directory can go missing — and when it does, Docker helpfully recreates
it empty and root-owned, which is what broke bootstrap and prompted all of
this. A file inside the package ships in the image and the wheel and has no
such failure mode.

Keeping a seed current is a deliberate step, not automatic: regenerate the
rules corpus with `python -m riftbound_bot.ingest.rules_export -o
<this dir>/core_rules_zh_tw.md` (re-adding the comment header it strips) and
commit the result.
"""
from __future__ import annotations

import json
from importlib.resources import files

RULES_SEED = "core_rules_zh_tw.md"
CARDS_SEED = "cards.json"


def rules_markdown() -> str:
    return files(__package__).joinpath(RULES_SEED).read_text(encoding="utf-8")


def cards() -> list[dict]:
    return json.loads(files(__package__).joinpath(CARDS_SEED).read_text(encoding="utf-8"))
