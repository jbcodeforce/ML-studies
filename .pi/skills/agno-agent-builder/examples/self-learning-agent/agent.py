"""
Self-Learning Agent

An agno agent that self-improves over time by saving reusable insights
to a knowledge base (ChromaDb) with human-in-the-loop confirmation.

Demonstrates:
* Knowledge base integration (ChromaDb + OllamaEmbedder)
* Human-in-the-loop tool approval (requires_confirmation=True)
* Semantic search for prior learnings
* Structured workflow with context awareness

Run with: python agent.py
Requires: ollama serve running (llama3.2 or mistral:7b-instruct)
"""

import json
from datetime import datetime, timezone

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools import tool
from agno.db.sqlite import SqliteDb
from agno.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.reader.text_reader import TextReader
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_LLM_TEMPERATURE = 0.4

SKILL_DIR = __import__("pathlib").Path(__file__).resolve().parent
TMP_DIR = SKILL_DIR / "tmp"
MEMORY_DB_PATH = TMP_DIR / "self-learning-memory.db"
KNOWLEDGE_PATH = TMP_DIR / "chromadb"

MAX_MEMORY_SIZE = 200  # max entries in knowledge base


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
INSTRUCTIONS = """\
You are a Self-Learning Agent that improves over time.

You have two special abilities:
1. Search your knowledge base for previously saved learnings
2. Save new insights using the save_learning tool (requires human approval)

## Workflow

1. **Check Knowledge First**
   - Before answering, always search for relevant prior learnings
   - Apply any matching insights to your response
   - If nothing relevant is found, proceed normally

2. **Gather Information**
   - Use any available tools to gather what you need
   - Combine tool results with your knowledge base insights

3. **Save Valuable Insights**
   - After useful interactions, propose saving a learning
   - The user will be asked to confirm before it's saved
   - Good learnings are specific, actionable, and generalizable

## What Makes a Good Learning

- **Specific**: "Tech P/E ratios typically range 20-35x" not "P/E varies"
- **Actionable**: Can be applied to future questions directly
- **Reusable**: Useful beyond this one conversation

### Good Examples
- "Our CI pipeline fails on Python 3.11 due to setuptools version"
- "The logger uses set-level(), not setLevel()"
- "API rate limits are 100 req/min — scale requests for bulk operations"

### Bad Examples
- "P/E ratios vary" (too vague)
- "Python is a programming language" (obvious)
- Raw data dumps (not reusable insight)

## Rules

- Never save data that is just raw output from a tool
- Don't save things that are obvious or trivially searchable
- Keep each learning to 1-3 sentences
- Include enough context to understand WHEN and WHY the learning applies
- Always credit the domain/source when relevant
"""


# ---------------------------------------------------------------------------
# Persistent Memory: SQLite
# ---------------------------------------------------------------------------
agent_db = SqliteDb(db_file=str(MEMORY_DB_PATH))


# ---------------------------------------------------------------------------
# Knowledge Base: ChromaDb with Semantic Search
# ---------------------------------------------------------------------------
learnings_kb = Knowledge(
    name="Agent Learnings",
    description="Reusable insights and patterns the agent learns over time",
    vector_db=ChromaDb(
        name="learnings",
        collection="learnings",
        path=str(KNOWLEDGE_PATH),
        persistent_client=True,
        search_type=SearchType.hybrid,
        hybrid_rrf_k=60,
        embedder=OllamaEmbedder(id=DEFAULT_LLM_MODEL, dimensions=3072),
    ),
    max_results=5,
    contents_db=agent_db,
)


# ---------------------------------------------------------------------------
# Custom Tool: Save Learning (human-in-the-loop)
# ---------------------------------------------------------------------------
@tool(requires_confirmation=True)
def save_learning(title: str, learning: str, source: str = "conversation") -> str:
    """
    Save a reusable insight to the knowledge base for future reference.

    This tool requires human confirmation before saving.
    The user will be prompted to approve or reject.

    Args:
        title: Short descriptive title (e.g., "CI pipeline Python 3.11 issue")
        learning: The insight to save — be specific and actionable
        source: Where this learning came from (e.g., "conversation", "document", "bug-report")

    Returns:
        Confirmation message
    """
    # Validate inputs
    if not title or not title.strip():
        return "Cannot save: title is required"
    if not learning or not learning.strip():
        return "Cannot save: learning content is required"

    # Build the payload
    payload = {
        "title": title.strip(),
        "learning": learning.strip(),
        "source": source or "conversation",
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save to knowledge base
    try:
        learnings_kb.insert(
            name=payload["title"],
            text_content=json.dumps(payload, ensure_ascii=False),
            reader=TextReader(),
            skip_if_exists=True,
        )

        # Also store in memory DB for structured access
        agent_db.insert("learnings.json", json.dumps(payload))

        return f"Saved learning: '{title}' (source: {source or 'conversation'})"
    except Exception as e:
        return f"Error saving learning: {e}"


# ---------------------------------------------------------------------------
# Agent Definition
# ---------------------------------------------------------------------------
learning_agent = Agent(
    name="Self-Learning Agent",
    model=Ollama(id=DEFAULT_LLM_MODEL, temperature=DEFAULT_LLM_TEMPERATURE),
    instructions=INSTRUCTIONS,
    tools=[
        save_learning,
    ],
    db=agent_db,
    knowledge=learnings_kb,
    search_knowledge=True,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
)


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    console = Console()

    # Show current knowledge base size
    try:
        from agno.db.sqlite import SqliteDb
        mem_db = SqliteDb(db_file=str(MEMORY_DB_PATH))
        count = mem_db.count_records("learnings.json") if mem_db else 0
    except Exception:
        count = 0

    console.print(Panel(
        f"[bold green]Self-Learning Agent[/bold green]\n"
        f"[dim]Learnings stored: {count}[/dim]\n"
        f"[dim]Knowledge base: {learnings_kb.name}[/dim]",
        title="[bold]Self-Learning Agent[/bold]",
    ))

    done = False
    agent = learning_agent
    while not done:
        question = Prompt.ask("Question >", default="")
        if not question or "bye" in question.lower():
            done = True
            continue

        console.print(f"[cyan]→ {question}[/cyan]")

        try:
            run_response = agent.run(question)

            if run_response.content:
                content_str = run_response.content
                if hasattr(content_str, "model_dump"):
                    content_str = json.dumps(
                        content_str.model_dump(),
                        indent=2,
                        ensure_ascii=False,
                    )
                console.print(content_str)

            # Handle active requirements (confirmation for save_learning)
            if run_response.active_requirements:
                console.print("\n[bold yellow]✋ Confirmation Required[/bold yellow]")
                console.print(f"  Tool: {run_response.requirements[0].tool_execution.tool_name}")
                console.print(f"  Args: {run_response.requirements[0].tool_execution.tool_args}")

                choice = Prompt.ask(
                    "Do you want to continue?",
                    choices=["y", "n"],
                    default="y",
                ).strip().lower()

                if choice == "n":
                    run_response.requirements[0].reject()
                    console.print("[red]Rejected — learning not saved[/red]")
                else:
                    run_response.requirements[0].confirm()
                    console.print("[green]Approved — learning saved![/green]")

                # Continue with the run
                run_response = agent.continue_run(
                    run_id=run_response.run_id,
                    requirements=run_response.requirements,
                )
                if run_response.content:
                    console.print(run_response.content)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("\n[dim]Goodbye![/dim]")
