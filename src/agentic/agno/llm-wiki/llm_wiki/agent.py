"""Assemble the LLM Wiki Agno agent."""

from __future__ import annotations

import os
from pathlib import Path

from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.models.ollama import Ollama

from llm_wiki.paths import WikiPaths, ensure_layout
from llm_wiki.prompt import load_wiki_instructions
from llm_wiki.stack import create_dbs, create_knowledge
from llm_wiki.tools import build_wiki_tools


def create_wiki_agent(
    paths: WikiPaths,
    *,
    rules_path: Path | None = None,
    model_id: str | None = None,
    debug_mode: bool = False,
) -> tuple[Agent, Knowledge]:
    ensure_layout(paths)
    contents_db, agent_db = create_dbs(paths)
    knowledge = create_knowledge(paths, contents_db)
    tools = build_wiki_tools(paths, knowledge)
    instructions = load_wiki_instructions(rules_path)

    mid = model_id or os.environ.get("LLM_WIKI_MODEL", "gemma4:26b")
    agent = Agent(
        name="LLM Wiki",
        description="Karpathy-style wiki maintainer with Agno persistence",
        model=Ollama(id=mid),
        instructions=instructions,
        tools=tools,
        knowledge=knowledge,
        search_knowledge=True,
        db=agent_db,
        markdown=True,
        num_history_runs=8,
        add_history_to_context=True,
        add_datetime_to_context=True,
        debug_mode=debug_mode,
    )
    return agent, knowledge


system_prompt = """
## Purpose

This wiki is a structured, interlinked knowledge base.
You maintain the wiki. The human curates sources, asks questions, and guides the analysis.

## Folder structure
```
wiki/         -- markdown pages maintained by Claude
wiki/index.md -- table of contents for the entire wiki
wiki/log.md   -- append-only record of all operations
```

## Ingest workflow

For the file path specified:

1. Read the full source document
2. Discuss key takeaways with the user before writing anything
3. Create a summary page in `wiki/` named after the source
4. Create or update concept pages for each major idea or entity
5. Add wiki-links ([[page-name]]) to connect related pages
6. Update `wiki/index.md` with new pages and one-line descriptions
7. Append an entry to `wiki/log.md` with the date, source name, and what changed


A single source may touch 10-15 wiki pages. That is normal.

## Page format

Every wiki page should follow this structure:


```markdown
# Page Title


**Summary**: One to two sentences describing this page.


**Sources**: List of raw source files this page draws from.


**Last updated**: Date of most recent update.

---


Main content goes here. Use clear headings and short paragraphs.


Link to related concepts using [[wiki-links]] throughout the text.


## Related pages


- [[related-concept-1]]
- [[related-concept-2]]
```


## Citation rules

- Every factual claim should reference its source file
- Use the format (source: filename.pdf) after the claim
- If two sources disagree, note the contradiction explicitly
- If a claim has no source, mark it as needing ve
"""

def create_wiki_agent_for_batch_ingestion(
    paths: WikiPaths,
    *,
    model_id: str | None = None,
    debug_mode: bool = False,
) -> Agent:
    ensure_layout(paths)
    contents_db, agent_db = create_dbs(paths)
    knowledge = create_knowledge(paths, contents_db)
    tools = build_wiki_tools(paths, knowledge)
    mid = model_id or os.environ.get("LLM_WIKI_MODEL", "gemma4:26b")
    agent = Agent(
        name="LLM Wiki",
        description="Karpathy-style wiki for batch ingestion",
        model=Ollama(id=mid),
        instructions=system_prompt,
        tools=tools,
        db=agent_db,
        markdown=True,
        num_history_runs=8,
        add_history_to_context=True,
        add_datetime_to_context=True,
        debug_mode=debug_mode,
    )
    return agent