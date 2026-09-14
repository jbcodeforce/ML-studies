"""
QA Agent (Quality Assurance code agent)

A software quality-assurance agent built with Agno and the local oMLX LLM
server (OpenAI-compatible).  It follows the `agno-agent-builder` skill:
custom local tools, a structured system prompt, persistent memory
(SQLite) and an optional semantic knowledge base.

Capabilities
------------
* analyze_code     -> static analysis of a source file (heuristics)
* run_tests        -> execute a test command and summarise results
* document_issue   -> persist a structured defect/finding
* search_issues    -> keyword search over past issues
* save_insight     -> persist a reusable insight / learning
* search_insights  -> keyword search over past insights

Persistence
-----------
* `tmp/qa-memory.db`          -> agno SqliteDb (session / learning history)
* `tmp/qa_agent.db`           -> raw records: issues + insights (sqlite3)
* `tmp/chromadb/`             -> optional semantic knowledge base
                               (only built when EMBEDDER_MODEL is provided)

Model
-----
Talks to an OpenAI-compatible server (oMLX by default at
http://127.0.0.1:7999/v1).  Configuration is environment-driven so it
matches the rest of this project:

    LLM_BASE_URL     -> http://127.0.0.1:7999/v1
    LLM_MODEL        -> e.g. Qwen3.8-27B-4bit
    LLM_API_KEY      -> local-key
    LLM_TEMPERATURE  -> 0.4
    EMBEDDER_MODEL   -> optional model id for semantic search (unset by default)
    EMBEDDER_BASE_URL     -> http://127.0.0.1:7999/v1 or http://10.0.0.148:7999/v1
Interactive use:
    python qa_agent.py
    python qa_agent.py --query "Analyze tests/ and report coverage"

Structured one-shot report:
    python qa_agent.py --query "..." --structured

Run in this project: source ../set_env.sh with ML_ENV_FILE=.env first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai.like import OpenAILike
from agno.tools import tool
from agno.tools.file import FileTools
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------
load_dotenv()

_env_file = os.getenv("ML_ENV_FILE")
if _env_file:
    load_dotenv(_env_file)
else:
    load_dotenv()

CODE_ROOT = Path(__file__).resolve().parent
TMP_DIR = CODE_ROOT / "tmp"
MEMORY_DB_PATH = TMP_DIR / "qa-memory.db"          # agno session memory
RECORD_DB_PATH = TMP_DIR / "qa_agent.db"          # raw issue/insight records
RECORD_LOCK = threading.Lock()                     # serialise sqlite writes
KNOWLEDGE_PATH = TMP_DIR / "chromadb"             # optional knowledge base

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:7999/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.8-27B-4bit")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "local-key")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL")       # optional -> enables KB
EMBEDDER_BASE_URL = os.getenv("EMBEDDER_BASE_URL", DEFAULT_LLM_BASE_URL)


def _ts() -> str:
    """ISO-8601 UTC timestamp (seconds)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _iso() -> str:
    """ISO-8601 UTC timestamp (with microseconds)."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Persistent record store (raw SQLite via stdlib)
# ---------------------------------------------------------------------------
class QARecordStore:
    """Lightweight, dependency-free store for QA issues and insights."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS issues (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    NOT NULL,
                severity   TEXT    NOT NULL DEFAULT 'Minor',
                status     TEXT    NOT NULL DEFAULT 'Open',
                file       TEXT    NOT NULL,
                line       INTEGER NOT NULL DEFAULT 0,
                description TEXT    NOT NULL,
                recommendation TEXT NOT NULL,
                created_at TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_issues_search
                ON issues (title, severity, file, description, recommendation);

            CREATE TABLE IF NOT EXISTS insights (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_insights_search
                ON insights (title, content);
            """
        )
        conn.commit()

    def add_issue(self, title: str, severity: str, file: str, line: int,
                  description: str, recommendation: str) -> int:
        with RECORD_LOCK, self._connect() as conn:
            self._init(conn)
            cur = conn.execute(
                "INSERT INTO issues "
                "(title, severity, status, file, line, description, recommendation, created_at) "
                "VALUES (?, ?, 'Open', ?, ?, ?, ?, ?)",
                (title, severity, file, line, description, recommendation, _iso()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def add_insight(self, title: str, content: str) -> int:
        with RECORD_LOCK, self._connect() as conn:
            self._init(conn)
            cur = conn.execute(
                "INSERT INTO insights (title, content, created_at) VALUES (?, ?, ?)",
                (title, content, _iso()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def search_issues(self, query: str, limit: int = 20) -> list[dict]:
        like = f"%{query}%"
        with RECORD_LOCK, self._connect() as conn:
            self._init(conn)
            rows = conn.execute(
                "SELECT id, title, severity, status, file, line, "
                "       description, recommendation, created_at "
                "FROM issues "
                "WHERE title LIKE ? OR severity LIKE ? OR file LIKE ? OR "
                "      description LIKE ? OR recommendation LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (like, like, like, like, like, limit),
            ).fetchall()
            print(rows)
        return [dict(zip(
            ["id", "title", "severity", "status", "file", "line",
             "description", "recommendation", "created_at"],
            row,
        )) for row in rows]

    def search_insights(self, query: str, limit: int = 20) -> list[dict]:
        like = f"%{query}%"
        with RECORD_LOCK, self._connect() as conn:
            self._init(conn)
            rows = conn.execute(
                "SELECT id, title, content, created_at FROM insights "
                "WHERE title LIKE ? OR content LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (like, like, limit),
            ).fetchall()
        return [dict(zip(["id", "title", "content", "created_at"], row)) for row in rows]


record_store = QARecordStore(RECORD_DB_PATH)


# ---------------------------------------------------------------------------
# Pydantic schemas (structured one-shot report)
# ---------------------------------------------------------------------------
class Issue(BaseModel):
    """A single QA finding."""

    title: str = Field(..., description="Short title of the issue")
    severity: str = Field(..., description="Critical / Major / Minor / Info")
    status: str = Field(..., description="Pass / Fail / Blocker / N/A / Open")
    file: str = Field(..., description="File path where the issue was found")
    line: int = Field(default=0, description="Line number (0 if N/A)")
    description: str = Field(..., description="Detailed description of the issue")
    reproduction_steps: str = Field(default="", description="Steps to reproduce (if applicable)")
    recommendation: str = Field(..., description="Suggested fix or action")


class QAReport(BaseModel):
    """Full QA report with a summary and a list of findings."""

    project: str = Field(..., description="Project name being analyzed")
    generated_at: str = Field(default_factory=_iso, description="ISO timestamp of the report")
    tests_run: Optional[int] = Field(default=None, description="Total tests executed")
    tests_passed: Optional[int] = Field(default=None, description="Tests that passed")
    tests_failed: Optional[int] = Field(default=None, description="Tests that failed")
    coverage: Optional[float] = Field(default=None, description="Test coverage (0-100, if known)")
    issues: list[Issue] = Field(default_factory=list, description="Findings")
    summary: str = Field(..., description="One-line executive summary")


# ---------------------------------------------------------------------------
# Lightweight static analysis
# ---------------------------------------------------------------------------
_SECRETS = [
    (r"(?i)\b(password|passwd|pwd|api_key|apikey|token|secret|secret_key|aws_secret)\s*=\s*['\"][^'\"]+['\"]"),
    (r"(?i)\beyJ[A-Za-z0-9_/-]{10,}\b"),           # JWT-ish
    (r"(?i)\bBEGIN\s+RSA\s+PRIVATE\s+KEY\b"),
]

def _run_tests(command: str, timeout: int = 120) -> dict:
    """Execute a test command from the workspace and summarise the result."""
    cwd = os.environ.get("QA_AGENT_WORKSPACE") or os.getcwd()

    if timeout <= 0:
        return {"error": "timeout must be a positive number of seconds", "status": "invalid"}

    if not command.strip():
        return {"error": "no command provided", "status": "invalid"}

    completed = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")

    result: dict = {
        "status": "completed",
        "exit_code": completed.returncode,
        "timestamp": _ts(),
        "tail": output.strip().splitlines()[-80:] if output.strip() else [],
    }

    last = output.strip().splitlines()[-30:]
    lowered = " ".join(last).lower()
    m_pass = re.search(r"(\d+)\s+(?:passed|pass)", lowered)
    m_fail = re.search(r"(\d+)\s+(?:failed|fail|error|error|errored|skipped)", lowered)

    if m_pass:
        result["tests_passed"] = int(m_pass.group(1))
    if m_fail:
        result["tests_failed"] = int(m_fail.group(1))
    if completed.returncode == 0:
        result["summary"] = f"Tests passed (exit code {completed.returncode})."
    else:
        result["summary"] = (f"Tests reported failures / non-zero exit ({completed.returncode}); "
                             f"see `tail` for details.")
    return result


# ---------------------------------------------------------------------------
# Custom Agno tools
# ---------------------------------------------------------------------------
@tool
def analyze_code(file_path: str, max_lines: int = 400) -> str:
    """
    Analyze a source file for bugs, TODOs and common code-smell patterns.

    Args:
        file_path: Path to the file to analyze (relative to workspace, or absolute)
        max_lines: Stop analysis after this many lines (0 = no limit)

    Returns:
        JSON summary with total line count, function count and any smells.
    """
    path = Path(file_path)
    if not path.exists():
        return {
            "error": f"File not found: {file_path}",
            "total_lines": 0, "functions": 0, "smells": [],
        }

    try:
        lines = path.read_text(errors="replace").split("\n")
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"Could not read file: {exc}", "total_lines": 0, "functions": 0, "smells": []}

    findings: list[str] = []
    func_count = 0
    lines_scanned = 0

    for i, raw in enumerate(lines, start=1):
        stripped = raw.strip()

        if i > max_lines:
            findings.append(
                f"Line {i}: file exceeds {max_lines} lines (truncated); large files need focused review.")
            break

        if re.search(r"\bdef\b.*\(|\basync\s+def\b.*\(", stripped):
            func_count += 1

        upper = stripped.upper()
        for marker in ("TODO", "FIXME", "XXX", "HACK", "HACKED"):
            if marker in upper:
                findings.append(f"Line {i}: {marker} marker present.")
                break

        if stripped.startswith("except") and ":" in stripped and "except" in stripped and "as" not in upper:
            # bare 'except:' (heuristic: no 'as' alias)
            if re.search(r"\bexcept\s*:", raw):
                findings.append(f"Line {i}: bare 'except:' hides error context.")

        if re.search(r"\beval\s*\(", stripped) or re.search(r"\bexec\s*\(", stripped):
            findings.append(f"Line {i}: use of {stripped.split('(')[0].strip()}() is dangerous.")

        for pat in _SECRETS:
            if re.search(pat, raw):
                findings.append(f"Line {i}: possible hardcoded secret.")
                break

        if ".execute" in stripped or "query" in stripped:
            if 'f"' in raw or "f'" in raw:
                findings.append(f"Line {i}: possible SQL injection (f-string in query).")

        if re.search(r"=\s*None\b|=\s*None$", raw) and not raw.startswith("#"):
            if "==" in raw or "!=" in raw:
                findings.append(f"Line {i}: compare with 'is'/\'is not\' instead of '== None'/\'!= None\'?")

        lines_scanned += 1

    info = {
        "file": str(path),
        "total_lines": len(lines),
        "functions": func_count,
        "lines_scanned": lines_scanned,
        "smells": findings,
    }
    return json.dumps(info, indent=2, ensure_ascii=False)


@tool
def run_tests(command: str = "pytest -qvs", timeout: int = 120) -> str:
    """
    Run a test command from the workspace and summarise the results.

    Args:
        command: Shell command to run (default: 'pytest -q')
        timeout: Maximum runtime in seconds

    Returns:
        JSON report (exit code, parsed pass/fail counts, and the tail of output).
    """
    data = _run_tests(command, timeout)
    return json.dumps(data, indent=2, ensure_ascii=False)


@tool
def document_issue(title: str, severity: str, status: str = "Open",
                   file: str = "", line: int = 0, description: str = "",
                   recommendation: str = "") -> str:
    """
    Persist a structured defect/finding.

    Args:
        title: Short descriptive title
        severity: One of Critical / Major / Minor / Info
        status: Open / Verified / Duplicate / N/A / Blocker
        file: Source file where the issue was found
        line: Line number (0 if N/A)
        description: Detailed description
        recommendation: Suggested fix or action

    Returns:
        JSON confirmation with the stored issue id.
    """
    issue_id = record_store.add_issue(
        title=title, severity=severity, file=file or "N/A", line=line,
        description=description, recommendation=recommendation,
    )
    return json.dumps(
        {"status": "documented", "issue_id": issue_id, "title": title,
         "severity": severity, "stored_at": _ts()},
        indent=2,
    )


@tool
def search_issues(query: str = "", limit: int = 20) -> str:
    """
    Search past QA issues by keyword.

    Args:
        query: Free-text keyword (title, severity, file, description, recommendation)
        limit: Maximum number of results

    Returns:
        JSON list of matching issues.
    """
    issues = record_store.search_issues(query or "*", limit)
    return json.dumps({"query": query, "count": len(issues), "issues": issues}, indent=2)


@tool
def save_insight(title: str, content: str) -> str:
    """
    Save a reusable insight or finding for future reference.

    Args:
        title: Short title for the insight
        content: Free-text insight content

    Returns:
        JSON confirmation with the stored insight id.
    """
    insight_id = record_store.add_insight(title=title, content=content)
    return json.dumps({"status": "saved", "insight_id": insight_id, "title": title}, indent=2)


@tool
def search_insights(query: str = "", limit: int = 20) -> str:
    """
    Search past saved insights by keyword.

    Args:
        query: Free-text keyword
        limit: Maximum number of results

    Returns:
        JSON list of matching insights.
    """
    insights = record_store.search_insights(query or "*", limit)
    return json.dumps({"query": query, "count": len(insights), "insights": insights}, indent=2)


# ---------------------------------------------------------------------------
# Optional semantic knowledge base (only if an embedder is configured)
# ---------------------------------------------------------------------------
def build_knowledge_base() -> Optional[object]:
    """
    Return a semantic knowledge base if a local embedding model is configured,
    otherwise None (the keyword tools cover persistence either way).
    """
    if not EMBEDDER_MODEL:
        return None

    from agno.knowledge import Knowledge
    from agno.knowledge.embedder.openai import OpenAIEmbedder
    from agno.knowledge.reader.text_reader import TextReader
    from agno.vectordb.chroma import ChromaDb
    from agno.vectordb.search import SearchType

    embedder = OpenAIEmbedder(id=EMBEDDER_MODEL, base_url=EMBEDDER_BASE_URL, api_key=DEFAULT_LLM_API_KEY)
    kb = Knowledge(
        name="QA Knowledge Base",
        description="Past QA findings, patterns, and project context",
        vector_db=ChromaDb(
            name="qa-knowledge",
            collection="qa-facts",
            path=str(KNOWLEDGE_PATH),
            persistent_client=True,
            search_type=SearchType.hybrid,
            hybrid_rrf_k=60,
            embedder=embedder,
        ),
        max_results=3,
        contents_db=SqliteDb(db_file=str(MEMORY_DB_PATH)),
    )
    return kb


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------
def _instructions(use_kb: bool) -> str:
    root = CODE_ROOT.parent  # parent of tmp/ qa_agent.py lives here
    lines = [
        "You are a Software Quality Assurance (QA) Agent — an automated testing",
        "and code-quality specialist. You analyze source code, run test suites, and",
        "report defects with clear severity, status and reproduction steps.",
        "",
        "## Workflow",
        "1. Understand the request — what is being tested or reviewed.",
        "2. Gather context — read relevant source files and search your knowledge.",
        "3. Execute — run existing tests and run static analysis heuristics.",
        "4. Report — document findings and suggest fixes.",
        "",
        "## Issue Severity Levels",
        "- Critical: crash, data loss, security vulnerability.",
        "- Major: core functionality broken or a required feature missing.",
        "- Minor: edge-case failures, inefficiencies, style issues.",
        "- Info: documentation / suggestion improvements.",
        "",
        "## Reporting Rules",
        "- Always assign a severity (Critical / Major / Minor / Info) and a status.",
        "- Give an exact file:line reference when the location is known.",
        "- Include reproduction steps for failing behaviour.",
        "- If nothing is wrong, say so explicitly.",
        "- Prefer concrete, actionable fixes. Use `document_issue` to persist findings.",
        "",
        f"## Workspace root: {root}",
    ]
    if use_kb:
        lines += [
            "",
            "## Knowledge Base",
            "- Before answering, search the knowledge base for related past findings.",
            "- Synthesize prior findings into your report; do not merely repeat them.",
        ]
    else:
        lines += [
            "",
            "## Note on memory",
            "- No semantic knowledge base is configured for this run. Persist new",
            "- findings with `document_issue` / `save_insight` and use `search_issues`",
            "- / `search_insights` to retrieve them.",
        ]
    return "\n".join(lines)


def build_agent(*, workspace: Path) -> Agent:
    """Construct the QA agent for a given workspace."""
    agent_db = SqliteDb(db_file=str(MEMORY_DB_PATH))
    knowledge = build_knowledge_base()
    use_kb = knowledge is not None

    tools: list = [
        FileTools(
            base_dir=workspace,
            enable_read_file=True,
            enable_read_file_chunk=True,
            enable_list_files=True,
            enable_search_files=True,
            enable_search_content=True,
        ),
        analyze_code,
        run_tests,
        document_issue,
        search_issues,
        save_insight,
        search_insights,
    ]

    return Agent(
        name="QA Agent",
        model=OpenAILike(
            id=DEFAULT_LLM_MODEL,
            base_url=DEFAULT_LLM_BASE_URL,
            api_key=DEFAULT_LLM_API_KEY,
            temperature=DEFAULT_LLM_TEMPERATURE,
        ),
        instructions=_instructions(use_kb),
        tools=tools,
        db=agent_db,
        knowledge=knowledge,
        search_knowledge=use_kb,
        add_datetime_to_context=True,
        add_history_to_context=True,
        num_history_runs=5,
        markdown=True,
    )


def verify_model_available(*, model: str, base_url: str, api_key: str,
                           verbose: bool) -> bool:
    """Warn (or fail) if the requested model is not exposed by the server."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=5.0)
        resp.raise_for_status()
        models = {m.get("id") for m in resp.json().get("data", [])}
        if model not in models:
            if verbose:
                print(
                    f"[warning] Model '{model}' not found on {base_url}.",
                    file=sys.stderr,
                )
                print(f"[warning] Available: {sorted(models) or '(none)'}", file=sys.stderr)
        return True
    except httpx.HTTPError as exc:
        if verbose:
            print(
                f"[warning] Could not reach the LLM server at {base_url}: {exc}.",
                file=sys.stderr,
            )
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_header() -> None:
    console = Console()
    table = Table(title="[bold green]QA Agent[/bold green] (agno + oMLX)", show_header=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Description")
    for name in ("FileTools", "analyze_code", "run_tests", "document_issue",
                 "search_issues", "save_insight", "search_insights"):
        table.add_row(name, "Built-in capability")
    console.print(table)


def _run_once(question: str, console: Console, structured: bool, workspace: Path,
              model: str, base_url: str, api_key: str, verbose: bool) -> int:
    if not verify_model_available(model=model, base_url=base_url, api_key=api_key, verbose=verbose):
        print("[error] No reachable LLM server. Start it (e.g. the oMLX server) and retry.",
              file=sys.stderr)
        return 1

    agent = build_agent(workspace=workspace)

    if structured:
        try:
            response = agent.run(question, output_schema=QAReport)
        except Exception as exc:
            print(f"[red]Structured run failed: {exc}[/red]")
            return 1
        if response.content is not None and hasattr(response.content, "model_dump"):
            print(json.dumps(response.content.model_dump(), indent=2, ensure_ascii=False))
        elif response.content:
            print(response.content)
        return 0

    agent.print_response(question, stream=True)
    return 0


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality Assurance code agent (agno + oMLX).")
    parser.add_argument("--workspace", "-w", type=Path,
                        default=CODE_ROOT,
                        help="Directory the agent may read (default: this project).")
    parser.add_argument("--model", "-m", default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
                        help=f"Model id (default: {DEFAULT_LLM_MODEL}).")
    parser.add_argument("--query", "-q", default=None,
                        help="Run a single question and exit.")
    parser.add_argument("--structured", action="store_true",
                        help="Return the answer as a structured QAReport (JSON) instead of prose.")
    parser.add_argument("--temperature", "-t", default=os.getenv("LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE),
                        help="Model temperature (default: %(default)s).")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
                        help="OpenAI-compatible server URL (default: %(default)s).")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY),
                        help="API key for the server (default: %(default)s).")
    parser.add_argument("--no-model-check", action="store_true",
                        help="Skip the startup model-availability check.")
    parser.add_argument("--interactive", action="store_true",
                        help="Force the interactive REPL.")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()

    if not workspace.is_dir():
        print(f"[error] Workspace directory does not exist: {workspace}", file=sys.stderr)
        return 2

    os.environ["QA_AGENT_WORKSPACE"] = str(workspace)

    if not args.no_model_check:
        print(f"[dim]Checking model {args.model!r} on {args.base_url}...[/dim]")
        verify_model_available(model=args.model, base_url=args.base_url,
                                api_key=args.api_key, verbose=False)

    print(Panel(
        "[bold green]QA Agent[/bold green] — code quality & testing\n"
        f"[dim]workspace: {workspace}\n"
        f"[dim]model:     {args.model}\n"
        f"[dim]kb:        {'enabled' if EMBEDDER_MODEL else 'keyword tools only'}",
        title="QA Agent",
    ))

    if args.query:
        return _run_once(args.query, Console(), structured=args.structured,
                         workspace=workspace, model=args.model,
                         base_url=args.base_url, api_key=args.api_key, verbose=True)

    # Interactive REPL
    _print_header()
    agent = build_agent(workspace=workspace)
    done = False
    while not done:
        try:
            question = Prompt.ask("QA Agent", default="Type 'exit' to quit.")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("exit", "quit", "bye"):
            done = True
            continue
        try:
            if args.structured:
                response = agent.run(question, output_schema=QAReport)
                if response.content is not None and hasattr(response.content, "model_dump"):
                    print(json.dumps(response.content.model_dump(), indent=2, ensure_ascii=False))
                elif response.content:
                    print(response.content)
            else:
                agent.print_response(question, stream=True)
        except KeyboardInterrupt:
            continue
        except Exception as exc:
            console = Console()
            console.print(f"[red]Error: {exc}[/red]")

    print("[dim]Goodbye![/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
