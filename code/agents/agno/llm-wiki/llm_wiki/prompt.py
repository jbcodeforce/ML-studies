"""Load system instructions from techno/claude/llm-wiki.md plus Agno runtime hints."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from llm_wiki.paths import default_rules_path


AGNO_RUNTIME_PREFIX = """\
## Agno runtime (tool use)

You run inside an Agno agent with tools and a vector knowledge base over wiki pages.

- **Never** modify or create files under `raw/` — it is read-only. Use tools to read sources from `raw/`.
- **Write** all wiki content under `wiki/` using the provided tools. Keep `wiki/index.md` and `wiki/log.md` updated per the rules above.
- After creating or updating a wiki page, call `index_wiki_file` so the knowledge base stays aligned with files on disk.
- Search the knowledge base when answering questions (`search_knowledge` is enabled); still follow the workflow of consulting `wiki/index.md` when appropriate.

"""


@lru_cache(maxsize=8)
def _read_rules_file(resolved_path: str) -> str:
    return Path(resolved_path).read_text(encoding="utf-8")


def load_wiki_instructions(rules_path: Path | None = None) -> str:
    p = (rules_path or default_rules_path()).expanduser().resolve()
    return AGNO_RUNTIME_PREFIX + "\n\n" + _read_rules_file(str(p))
