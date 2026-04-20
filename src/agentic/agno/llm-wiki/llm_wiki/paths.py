"""Filesystem layout for the Karpathy-style wiki under the llm-wiki directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def llm_wiki_root() -> Path:
    """Directory containing `raw/`, `wiki/`, and `data/` (parent of the `llm_wiki` package)."""
    return Path(__file__).resolve().parent.parent


def ml_studies_root() -> Path:
    """ML-studies repo root (parent of `src/`)."""
    return llm_wiki_root().parent.parent.parent.parent


def default_rules_path() -> Path:
    override = os.environ.get("LLM_WIKI_RULES_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return ml_studies_root() / "techno" / "claude" / "llm-wiki.md"


@dataclass(frozen=True)
class WikiPaths:
    root: Path
    raw: Path
    wiki: Path
    data: Path
    contents_db: Path
    agent_db: Path
    chroma_path: Path


def wiki_paths(root: Path | None = None) -> WikiPaths:
    base = root if root is not None else llm_wiki_root()
    data = base / "data"
    return WikiPaths(
        root=base.resolve(),
        raw=(base / "raw").resolve(),
        wiki=(base / "wiki").resolve(),
        data=data.resolve(),
        contents_db=(data / "contents.db").resolve(),
        agent_db=(data / "agent.db").resolve(),
        chroma_path=(data / "chroma").resolve(),
    )


def ensure_layout(paths: WikiPaths) -> None:
    paths.raw.mkdir(parents=True, exist_ok=True)
    paths.wiki.mkdir(parents=True, exist_ok=True)
    paths.data.mkdir(parents=True, exist_ok=True)
    index = paths.wiki / "index.md"
    if not index.exists():
        index.write_text(
            "# Wiki index\n\n"
            "_Pages appear here as you ingest sources and grow the wiki._\n\n"
            "## Pages\n\n",
            encoding="utf-8",
        )
    log = paths.wiki / "log.md"
    if not log.exists():
        log.write_text(
            "# Wiki log\n\n"
            "_Append-only record of ingest and maintenance operations._\n\n",
            encoding="utf-8",
        )
