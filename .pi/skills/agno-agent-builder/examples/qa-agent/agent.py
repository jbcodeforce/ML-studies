"""
Software Quality Assurance Agent

An agno agent that performs QA tasks:
- Analyze source code and dependencies
- Run and evaluate test suites
- Document and track issues with severity and status
- Learn from past findings via a knowledge base

Demonstrates:
* Custom local tools (@tool decorator)
* System prompt with structured workflow and rules
* Persistent memory via SqliteDb
* Knowledge base for semantic recall of past issues
* Structured output via Pydantic schema

Run with: python agent.py
Requires: ollama serve running (llama3.2 or mistral:7b-instruct)
"""

from datetime import datetime, timezone
import json
from pathlib import Path

from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools import tool
from agno.db.sqlite import SqliteDb
from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.reader.text_reader import TextReader
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

_env_file = os.getenv("ML_ENV_FILE")
load_dotenv(_env_file) if _env_file else load_dotenv()

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:7999/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.8-27B-4bit")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "local-key")
DEFAULT_EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "bge-small-en-v1.5-8bit")

CODE_DIR = Path(__file__).resolve().parent
TMP_DIR = CODE_DIR / "tmp"
MEMORY_DB_PATH = TMP_DIR / "qa-memory.db"
KNOWLEDGE_PATH = TMP_DIR / "chromadb"

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
INSTRUCTIONS = f"""\
You are a Software Quality Assurance Agent — an automated testing specialist
that enforce testing, verify coverage, find bugs, evaluate code quality, and learn from every project you analyze.

## Your Tools

You have these tools at your disposal:
- analyze_code: Read and analyze source code files
- check_tests: Run the test suite and report coverage
- document_issue: Log a bug finding with severity and status
- review_pr: Review pull request changes

## Workflow

1. **Understand the Request**
   - What is being tested or reviewed?
   - What are the acceptance criteria?

2. **Gather Context**
   - Check project structure and dependencies
   - Search your knowledge base for similar past issues
   - Read relevant source files

3. **Execute Tests**
   - Run existing tests with check_tests
   - Analyze code for bugs, security issues, or performance problems

4. **Report Findings**
   - Document each issue with severity, status, and reproduction steps
   - Save insights to the knowledge base for future reference

## Issue Severity Levels

- Critical: System crash, data loss, security vulnerability
- Major: Core functionality broken, missing critical feature
- Minor: UI issues, minor inefficiencies, edge case failures
- Info: Style improvements, documentation suggestions

## Reporting Rules

- Always include test case ID or file reference
- Status: Pass | Fail | Blocker | N/A | Unknown
- Provide reproduction steps for failing tests
- If no issue found, explicitly say "No issues detected"
- Be concise — one paragraph per issue

## What Makes a Good QA Finding

- Reproducible: Can be consistently reproduced
- Specific: References exact file/line/location
- Actionable: Clear steps to fix or reproduce

---
"""

# ---------------------------------------------------------------------------
# Pydantic Output Schema
# ---------------------------------------------------------------------------
class Issue(BaseModel):
    """A single QA issue finding."""
    title: str = Field(..., description="Short title of the issue")
    severity: str = Field(
        ...,
        description="Critical / Major / Minor / Info",
    )
    status: str = Field(
        ...,
        description="Pass / Fail / Blocker / N/A / Unknown",
    )
    file: str = Field(..., description="File path where the issue was found")
    line: int = Field(..., description="Line number (0 if N/A)")
    description: str = Field(
        ...,
        description="Detailed description of the issue",
    )
    reproduction_steps: str = Field(
        ...,
        description="Steps to reproduce the issue (if applicable)",
    )
    recommendation: str = Field(
        ...,
        description="Suggested fix or action",
    )


class TestReport(BaseModel):
    """Full test report with summary and issues."""
    project: str = Field(..., description="Project name being analyzed")
    date: str = Field(
        ...,
        description="ISO timestamp of the report",
    )
    tests_run: int = Field(..., description="Total tests executed")
    tests_passed: int = Field(..., description="Tests that passed")
    tests_failed: int = Field(
        ...,
        description="Tests that failed",
    )
    coverage: float = Field(
        default=0.0,
        description="Test coverage percentage (0-100)",
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="List of issues found during analysis",
    )
    summary: str = Field(
        ...,
        description="One-line executive summary",
    )


# ---------------------------------------------------------------------------
# Persistent Memory: SQLite
# ---------------------------------------------------------------------------
agent_db = SqliteDb(db_file=str(MEMORY_DB_PATH))


# ---------------------------------------------------------------------------
# Knowledge Base: Chroma + Semantic Search
# ---------------------------------------------------------------------------
qa_kb = Knowledge(
    name="QA Knowledge Base",
    description="Past QA findings, patterns, and project context",
    vector_db=ChromaDb(
        name="qa-knowledge",
        collection="qa-facts",
        path=str(KNOWLEDGE_PATH),
        persistent_client=True,
        search_type=SearchType.hybrid,
        hybrid_rrf_k=60,
        embedder=OpenAIEmbedder(id=DEFAULT_EMBEDDER_MODEL, 
                                base_url=DEFAULT_LLM_BASE_URL,
                                api_key=DEFAULT_LLM_API_KEY,
                                dimensions=392),
    ),
    max_results=3,
    contents_db=agent_db,
)

# ---------------------------------------------------------------------------
# Custom Tools
# ---------------------------------------------------------------------------

@tool
def analyze_code(file_path: str, language: str = "auto") -> str:
    """
    Analyze source code for bugs, security issues, and code quality problems.

    Args:
        file_path: Path to the file (relative to project root, or full path)
        language: Programming language (auto-detect if 'auto')

    Returns:
        Analysis of the code including any issues found
    """
    path = Path(file_path)
    if not path.exists():
        return f"ERROR: File not found: {file_path}"

    content = path.read_text(errors="replace")
    lines = content.split("\n")

    issues = []
    lines_analyzed = 0
    functions_found = 0
    issues_found = 0

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Count functions
        if any(kw in stripped for kw in ["def ", "async def "]):
            functions_found += 1

        # Flag TODO/FIXME comments
        if any(marker in stripped for marker in ["TODO", "FIXME", "HACK"]):
            issues_found += 1
            issues.append(f"Line {i}: Comment marker: {marker}")

        # Flag suspicious patterns
        if "eval(" in stripped and not stripped.startswith("#"):
            issues_found += 1
            issues.append(f"Line {i}: POTENTIAL SECURITY RISK: eval() call")

        lines_analyzed += 1

        # Truncate for performance — analyze first 500 lines + rest summary
        if lines_analyzed > 500:
            break

    result = {
        "file": str(path),
        "language": language,
        "total_lines": len(lines),
        "functions": functions_found,
        "issues": issues_found,
        "issues_found": issues,
        "preview": "\n".join(lines[:30]),
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


@tool
def check_tests(test_file: str = "tests") -> str:
    """
    Run the test suite and report results.

    Args:
        test_file: Path to test file or test directory (relative to project root)

    Returns:
        JSON report with pass/fail status, coverage, and output
    """
    project_root = CODE_DIR.parent.parent
    test_path = project_root / test_file
    test_path = str(test_path) if not test_path.is_absolute() else test_path

    # Try pytest first
    import subprocess

    cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short", "-q"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return json.dumps(
            {
                "status": "success",
                "output": result.stdout or result.stderr,
                "exit_code": result.returncode,
                "command": " ".join(cmd),
            },
            indent=2,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "timeout", "error": "Test run exceeded 60 seconds"}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


@tool
def document_issue(
    title: str,
    severity: str,
    file: str,
    line: int,
    description: str,
    reproduction_steps: str,
    recommendation: str,
    project: str = None,
) -> str:
    """
    Document a QA finding with metadata for tracking and retrieval.

    Args:
        title: Short descriptive title
        severity: Critical / Major / Minor / Info
        file: Source file where the issue was found
        line: Line number (0 if not applicable)
        description: Detailed description of the issue
        reproduction_steps: Steps to reproduce
        recommendation: Suggested fix
        project: Project name (optional, auto-detected if not provided)
    """
    payload = {
        "title": title,
        "severity": severity,
        "file": file,
        "line": line,
        "description": description,
        "reproduction_steps": reproduction_steps,
        "recommendation": recommendation,
        "project": project or CODE_DIR.parent.parent.name,
        "documented_at": datetime.now(timezone.utc).isoformat(),
    }

    json_str = json.dumps(payload, ensure_ascii=False)

    # Store in memory DB
    try:
        agent_db.insert("qa_issues.json", json_str)
    except Exception:
        pass

    # Also store in knowledge base for semantic search
    try:
        qa_kb.insert(
            name=f"[{severity}] {title}",
            text_content=json_str,
            reader=TextReader(),
            skip_if_exists=True,
        )
    except Exception:
        pass

    return json.dumps(
        {
            "status": "documented",
            "title": title,
            "severity": severity,
            "stored_at": payload["documented_at"],
            "message": f"Issue documented: '{title}' [{severity}]",
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Agent Definition
# ---------------------------------------------------------------------------
qa_agent = Agent(
    name="QA Agent",
    model=OpenAILike(id=DEFAULT_LLM_MODEL, 
                     base_url=DEFAULT_LLM_BASE_URL,
                    api_key=DEFAULT_LLM_API_KEY,
                    temperature=DEFAULT_LLM_TEMPERATURE),
    instructions=INSTRUCTIONS,
    tools=[
        analyze_code,
        check_tests,
        document_issue,
    ],
    db=agent_db,
    knowledge=qa_kb,
    search_knowledge=True,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
    output_schema=TestReport,
)


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    console = Console()

    console.print(Panel(
        "[bold green]QA Agent[/bold green] — Software Quality Assurance\n"
        "[dim]Tools: analyze_code, check_tests, document_issue[/dim]\n"
        "[dim]Knowledge base enabled for semantic recall[/dim]",
        title="[bold]QA Agent v1.0[/bold]",
    ))

    done = False
    agent = qa_agent
    while not done:
        question = input("\nQuestion > ").strip()
        if not question or question.lower() in ("bye", "exit", "quit"):
            done = True
            continue

        console.print(f"[cyan]{question}[/cyan]")

        try:
            response = agent.run(question)

            if response.content:
                content_str = response.content
                if hasattr(response.content, "model_dump"):
                    # Pydantic output
                    content_str = json.dumps(
                        response.content.model_dump(),
                        indent=2,
                        ensure_ascii=False,
                    )
                console.print(content_str)

            # Handle active requirements (e.g., confirmation for document_issue)
            if response.active_requirements:
                console.print("\n[bold yellow]Confirmation Required[/bold yellow]")
                for req in response.active_requirements:
                    console.print(f"  Tool: {req.tool_execution.tool_name}")
                    console.print(f"  Args: {req.tool_execution.tool_args}")
                choice = input("Continue? (y/n) ").strip().lower()
                if choice == "n":
                    for req in response.active_requirements:
                        req.reject()
                else:
                    for req in response.active_requirements:
                        req.confirm()

                response = agent.continue_run(
                    run_id=response.run_id,
                    requirements=response.requirements,
                )
                if response.content:
                    console.print(response.content)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("\n[dim]Goodbye![/dim]")
