"""Filesystem and knowledge-index tools for the wiki agent."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from agno.knowledge.knowledge import Knowledge
from agno.tools import tool

from llm_wiki.fsutil import safe_resolve_under
from llm_wiki.indexing import index_wiki_markdown
from llm_wiki.paths import WikiPaths


def build_wiki_tools(paths: WikiPaths, knowledge: Knowledge) -> list:
    raw_root = paths.raw
    wiki_root = paths.wiki

    @tool
    def list_raw_sources() -> str:
        """List file names in the raw/ folder (immutable sources)."""
        if not raw_root.exists():
            return "raw/ does not exist yet."
        names = sorted(p.name for p in raw_root.iterdir() if p.is_file())
        if not names:
            return "No files in raw/ yet."
        return "Files in raw/:\n" + "\n".join(f"- {n}" for n in names)


    @tool
    def read_given_file(filepath: str) -> str:
        """Read a single file from the given filepath."""
        path = Path(filepath)
        if not path.is_file():
            return f"Not found: {filepath}"
        return path.read_text(encoding="utf-8", errors="replace")

    @tool
    def read_raw_file(filename: str) -> str:
        """Read a single file from raw/ by file name (basename only, no paths)."""
        name = Path(filename).name
        if name != filename.strip():
            raise ValueError("Use basename only (no directories)")
        path = raw_root / name
        if not path.is_file():
            return f"Not found: raw/{name}"
        return path.read_text(encoding="utf-8", errors="replace")

    @tool
    def read_wiki_file(relative_path: str) -> str:
        """Read a markdown file under wiki/ using a path relative to wiki/ (e.g. index.md)."""
        path = safe_resolve_under(wiki_root, relative_path)
        if not path.is_file():
            return f"Not found: wiki/{relative_path}"
        return path.read_text(encoding="utf-8", errors="replace")

    @tool
    def write_wiki_file(relative_path: str, content: str) -> str:
        """Create or overwrite a file under wiki/. Paths are confined to wiki/ only."""
        path = safe_resolve_under(wiki_root, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote wiki/{relative_path} ({len(content)} chars)"

    @tool
    def append_wiki_log(entry: str) -> str:
        """Append a dated entry to wiki/log.md."""
        path = wiki_root / "log.md"
        line = f"\n## {date.today().isoformat()}\n\n{entry.strip()}\n"
        if path.is_file():
            path.write_text(path.read_text(encoding="utf-8") + line, encoding="utf-8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Wiki log\n" + line, encoding="utf-8")
        return "Appended entry to wiki/log.md"

    @tool
    def index_wiki_file(relative_path: str) -> str:
        """Chunk and embed a wiki markdown file into the vector knowledge base."""
        try:
            index_wiki_markdown(paths, knowledge, relative_path)
        except FileNotFoundError:
            return f"Cannot index: not a file: wiki/{relative_path}"
        except ValueError as e:
            return str(e)
        return f"Indexed wiki/{relative_path} into knowledge base"

    return [
        list_raw_sources,
        read_raw_file,
        read_wiki_file,
        write_wiki_file,
        append_wiki_log,
        index_wiki_file,
        read_given_file
    ]
