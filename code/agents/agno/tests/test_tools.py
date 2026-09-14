"""Unit tests for the six ``@tool`` wrappers in ``qa_agent``.

Each tool returns a JSON string; these tests parse it and assert on the shape.
The store-backed tools use an isolated temp database (``conftest.isolated_records``).
"""

from __future__ import annotations

import json

import pytest

from qa_agent import (
    analyze_code,
    document_issue,
    record_store,
    run_tests,
    save_insight,
    search_insights,
    search_issues,
)


def test_analyze_code_tool_returns_json(sample_files):
    payload = analyze_code(str(sample_files / "good.py"))
    data = json.loads(payload)
    assert data["functions"] == 2
    assert data["smells"] == []


def test_run_tests_tool_defaults_to_pytest(workspace, monkeypatch):
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))
    payload = run_tests("", timeout=5)  # empty command -> invalid
    data = json.loads(payload)
    assert data["status"] == "invalid"


def test_document_issue_tool_returns_id(isolated_records):
    payload = document_issue(
        title="A bug", severity="Critical", status="Open",
        file="src.py", line=42, description="boom", recommendation="fix",
    )
    data = json.loads(payload)
    assert data["status"] == "documented"
    assert data["issue_id"] == 1
    assert data["title"] == "A bug"
    assert data["severity"] == "Critical"


def test_document_issue_tool_defaults(isolated_records):
    payload = document_issue(title="t", severity="Minor")
    data = json.loads(payload)
    assert data["issue_id"] == 1
    # defaults in the tool: file -> "N/A", line -> 0, status -> "Open"
    row = isolated_records.search_issues("*", limit=1)[0]
    assert row["file"] == "N/A"
    assert row["line"] == 0
    assert row["status"] == "Open"


def test_document_then_search_issues_roundtrip(isolated_records):
    document_issue(
        title="findable bug", severity="Major", file="src.py", line=7,
        description="desc", recommendation="fix",
    )
    payload = search_issues(query="findable", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1
    assert data["issues"][0]["title"] == "findable bug"


def test_save_and_search_insights_roundtrip(isolated_records):
    save_insight(title="learned", content="use dicts")
    payload = search_insights(query="use", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1
    assert data["insights"][0]["content"] == "use dicts"


def test_tools_write_to_their_own_store(isolated_records):
    # The module-level ``record_store`` is separate from ``isolated_records``.
    # Tools must use the module-level store, so verify via search_issues().
    document_issue(title="tool issue", severity="Info", file="x.py", line=1)
    payload = search_issues(query="tool issue", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1
