"""
MLX Code Agent — a simple Claude Code-style agent using Agno + oMLX or mlx_lm.server.

Start oMLX first (see startoLMX.sh), load a model in the admin UI, then run:

    uv run python mlx_llm_code_agent.py
    CODE_AGENT_MODEL=Ornith-1.0-9B-6bit uv run python mlx_llm_code_agent.py

Example prompts:
- "List files in src/agentic/agno"
- "Read mlx_llm_code_agent.py and summarize it"
- "Create a branch feature/demo and add a hello.txt with today's date"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai.like import OpenAILike
from agno.tools.coding import CodingTools

DEFAULT_LLM_BASE_URL = os.getenv("CODE_AGENT_BASE_URL", "http://127.0.0.1:7999/v1")
DEFAULT_LLM_API_KEY = os.getenv("MLX_API_KEY", "local-key")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("CODE_AGENT_TEMPERATURE", "0.4"))

DEFAULT_LLM_MODEL = os.getenv("CODE_AGENT_MODEL", "Ornith-1.0-9B-6bit")
ALLOWED_MODELS = ("DreamFoundries--Agents-A1-8bit", "Ornith-1.0-9B-6bit")

CODE_AGENT_INSTRUCTIONS = """\
You are a coding assistant with filesystem and git tools. Work only inside the
configured workspace directory.

## Workflow

1. Explore — use `ls` to understand directory structure before making changes.
2. Read — always `read_file` before `edit_file` to see current contents.
3. Edit — prefer small `edit_file` changes over rewriting entire files with `write_file`.
4. Git — create a branch before edits; stage specific files; write clear commit messages.
   Call git commands one at a time (no shell chaining).

## Rules

- Stay within the workspace; do not access paths outside it.
- Use `read_file` offset/limit for large files.
- Use `write_file` for new files; use `edit_file` for modifications.
- Never modify `.env`, credentials, secrets, or `tmp/` databases unless explicitly asked.
- Report errors clearly; do not guess file contents.
"""


def resolve_model_id(model_id: str | None = None) -> str:
    """Return a validated model id from arg, env, or default."""
    mid = (model_id or DEFAULT_LLM_MODEL).strip()
    if mid not in ALLOWED_MODELS:
        allowed = ", ".join(ALLOWED_MODELS)
        raise ValueError(f"Unknown model '{mid}'. Allowed: {allowed}")
    return mid


def verify_model_available(model_id: str, base_url: str, api_key: str) -> None:
    """Warn if the model is not listed by the oMLX /v1/models endpoint."""
    models_url = base_url.rstrip("/") + "/models"
    try:
        resp = httpx.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        ids = {m.get("id") for m in data.get("data", [])}
        if model_id not in ids:
            print(
                f"[warning] Model '{model_id}' not found in {models_url}.\n"
                f"  Available: {sorted(ids) or '(none)'}\n"
                f"  Load the model in oMLX admin before running the agent.",
                file=sys.stderr,
            )
    except httpx.HTTPError as exc:
        print(
            f"[warning] Could not reach oMLX at {models_url}: {exc}\n"
            f"  Start oMLX with ./startoLMX.sh and load model '{model_id}'.",
            file=sys.stderr,
        )


def build_tools(workspace: Path) -> CodingTools:
    """Filesystem + git tools scoped to workspace."""
    return CodingTools(
        base_dir=workspace,
        restrict_to_base_dir=True,
        enable_ls=True,
        enable_read_file=True,
        enable_write_file=True,
        enable_edit_file=True,
        enable_run_shell=True,
    )


def create_code_agent(
    *,
    workspace: Path | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Agent:
    """Create the MLX code agent."""
    workspace = (workspace or Path.cwd()).resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace does not exist: {workspace}")

    mid = resolve_model_id(model_id)
    url = base_url or DEFAULT_LLM_BASE_URL
    key = api_key or DEFAULT_LLM_API_KEY

    return Agent(
        name="MLX Code Agent",
        model=OpenAILike(
            id=mid,
            base_url=url,
            api_key=key,
            temperature=DEFAULT_LLM_TEMPERATURE,
        ),
        tools=[build_tools(workspace)],
        db=SqliteDb(db_file="tmp/code_agent.db"),
        instructions=CODE_AGENT_INSTRUCTIONS,
        markdown=True,
        add_history_to_context=True,
        num_history_runs=10,
        add_datetime_to_context=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Code Agent (Agno + oMLX or mlx_lm.server)")
    parser.add_argument(
        "--workspace",
        "-w",
        type=Path,
        default=None,
        help="Workspace root for file/git tools (default: cwd)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help=f"MLX model id (default: {DEFAULT_LLM_MODEL})",
    )
    args = parser.parse_args()

    workspace = (args.workspace or Path.cwd()).resolve()
    model_id = resolve_model_id(args.model)

    verify_model_available(model_id, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_API_KEY)

    print(f"MLX Code Agent")
    print(f"  model:     {model_id}")
    print(f"  workspace: {workspace}")
    print(f"  MLX server:      {DEFAULT_LLM_BASE_URL}")
    print("Type a task, or empty line / 'bye' to quit.\n")

    agent = create_code_agent(workspace=workspace, model_id=model_id)

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("bye", "exit", "quit"):
            break
        agent.print_response(question, stream=True)
        print()


if __name__ == "__main__":
    main()
