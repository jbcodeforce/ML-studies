"""
Workspace tools to work locally: read/write/edit/search/shell access to a directory as referenced
by WORKSPACE_DIR env, var.
"""


from ..config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE, DEFAULT_LLM_API_KEY
import os
from pathlib import Path
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.workspace import Workspace
from agno.db.sqlite import SqliteDb

THIS_CODE_FOLDER = Path(__file__).resolve().parent

if os.getenv("WORKSPACE_DIR"):
    WORKSPACE_DIR=Path(os.getenv("WORKSPACE_DIR", "./workspace"))
else:
    WORKSPACE_DIR = THIS_CODE_FOLDER / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def init_workspace():
    (WORKSPACE_DIR / "README.md").write_text(
        "# Agent workspace"
    )

def build_agent() -> Agent:
    """
    workspace tools=['read_file', 'list_files', 'search_content', 'write_file', 'edit_file', 'move_file', 'delete_file', 'run_command']
    """
    model = OpenAILike(
        id=DEFAULT_LLM_MODEL,
        base_url=DEFAULT_LLM_BASE_URL,
        temperature=DEFAULT_LLM_TEMPERATURE,
        api_key=DEFAULT_LLM_API_KEY,
    )
    agent_db = SqliteDb(db_file=str(WORKSPACE_DIR / "agents.db"))
    agent = Agent(
        name="Agent",
        model=model,
        tools=[
                Workspace(
                    str(WORKSPACE_DIR),
                    allowed=Workspace.ALL_TOOLS,  # will run silently
                    confirm=[],
                    exclude_patterns=["agents.db"]
                )
            ],
        markdown=True,
    )
    return agent

if __name__ == "__main__":
    # prepare it
    init_workspace()
    agent = build_agent()
    agent.print_response("what are the files in workspace?")
    print(f"\nWorkspace: {WORKSPACE_DIR}")