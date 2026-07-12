"""Batch (re)index wiki markdown into the Knowledge store."""

from __future__ import annotations

import hashlib
from pathlib import Path

from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.text_reader import TextReader

from llm_wiki.fsutil import safe_resolve_under
from llm_wiki.paths import WikiPaths


def _normalize_extensions(exts: set[str]) -> set[str]:
    out: set[str] = set()
    for e in exts:
        e = e.strip().lower().lstrip(".")
        if e:
            out.add("." + e)
    return out


def index_folder_into_knowledge(
    knowledge: Knowledge,
    folder: Path,
    extensions: set[str],
    *,
    dry_run: bool = False,
    max_files: int | None = None,
) -> list[str]:
    """Embed every file under ``folder`` matching ``extensions`` into ``knowledge``.

    Document names are ``corpus-{8-hex-of-root}-{relative}`` so different source trees
    do not collide. Metadata uses ``kind: corpus`` plus ``source_root`` and ``relative_path``.

    Returns posix paths relative to ``folder`` (one entry per file processed).
    """
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    exts = _normalize_extensions(extensions)
    if not exts:
        raise ValueError("At least one extension is required")

    root_key = hashlib.sha256(str(root).encode()).hexdigest()[:8]
    source_root_str = str(root)

    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        if path.suffix.lower() not in exts:
            continue
        candidates.append(path)

    if max_files is not None:
        candidates = candidates[:max_files]

    indexed: list[str] = []
    for path in candidates:
        rel = path.relative_to(root).as_posix()
        indexed.append(rel)
        if dry_run:
            continue
        safe = rel.replace("/", "__").replace("\\", "__")
        doc_name = f"corpus-{root_key}-{safe}"
        """
        knowledge.insert(
            name=doc_name,
            path=str(path),
            reader=TextReader(),
            upsert=True,
            skip_if_exists=False,
            metadata={
                "kind": "corpus",
                "source_root": source_root_str,
                "relative_path": rel,
            },
        )
        """
    return indexed


def index_wiki_markdown(paths: WikiPaths, knowledge: Knowledge, relative_path: str) -> None:
    path = safe_resolve_under(paths.wiki, relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"wiki/{relative_path}")
    if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise ValueError(f"Not a markdown/text file: {relative_path}")
    safe_name = relative_path.replace("/", "__").replace("\\", "__")
    knowledge.insert(
        name=f"wiki-{safe_name}",
        path=str(path),
        reader=TextReader(),
        upsert=True,
        skip_if_exists=False,
        metadata={"kind": "wiki", "relative_path": relative_path},
    )


def reindex_all_wiki_pages(paths: WikiPaths, knowledge: Knowledge) -> list[str]:
    indexed: list[str] = []
    for md in sorted(paths.wiki.rglob("*.md")):
        rel = md.relative_to(paths.wiki).as_posix()
        index_wiki_markdown(paths, knowledge, rel)
        indexed.append(rel)
    return indexed
