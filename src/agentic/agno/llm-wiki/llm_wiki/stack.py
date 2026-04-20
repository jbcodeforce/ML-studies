"""SqliteDb + Chroma + Knowledge factories for the LLM wiki."""

from __future__ import annotations

import os

from agno.db.sqlite import SqliteDb
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType

from llm_wiki.paths import WikiPaths


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def create_dbs(paths: WikiPaths) -> tuple[SqliteDb, SqliteDb]:
    """Contents DB (knowledge metadata) and agent session DB."""
    contents_db = SqliteDb(
        db_file=str(paths.contents_db),
        knowledge_table="llm_wiki_kb",
    )
    agent_db = SqliteDb(db_file=str(paths.agent_db))
    return contents_db, agent_db


def create_knowledge(
    paths: WikiPaths,
    contents_db: SqliteDb,
    *,
    embedder_id: str | None = None,
    embedder_dimensions: int | None = None,
    max_results: int | None = None,
) -> Knowledge:
    eid = embedder_id or os.environ.get("LLM_WIKI_EMBEDDER", "nomic-embed-text")
    dims = embedder_dimensions if embedder_dimensions is not None else _env_int("LLM_WIKI_EMBED_DIM", 768)
    mr = max_results if max_results is not None else _env_int("LLM_WIKI_MAX_RESULTS", 8)

    embedder = OllamaEmbedder(id=eid, dimensions=dims)
    vector_db = ChromaDb(
        name="llm_wiki_vectors",
        collection="llm_wiki",
        path=str(paths.chroma_path),
        persistent_client=True,
        embedder=embedder,
        search_type=SearchType.hybrid,
        hybrid_rrf_k=60,
    )
    return Knowledge(
        name="LLM Wiki",
        description="Karpathy-style wiki markdown indexed for retrieval",
        vector_db=vector_db,
        contents_db=contents_db,
        max_results=mr,
    )
