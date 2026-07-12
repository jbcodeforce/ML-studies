"""Safe path resolution under a root directory."""

from __future__ import annotations

from pathlib import Path


def safe_resolve_under(root: Path, relative: str) -> Path:
    root_r = root.resolve()
    rel = relative.strip().replace("\\", "/").lstrip("/")
    if not rel:
        raise ValueError("Path must not be empty")
    for part in rel.split("/"):
        if part == "..":
            raise ValueError("Path must not contain '..'")
    full = (root_r / rel).resolve()
    full.relative_to(root_r)
    return full
