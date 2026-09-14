"""Unit tests for the persistence layer: ``qa_agent.QARecordStore``.

These use an isolated temp database (see ``conftest.isolated_records``) so the
project's real ``tmp/qa_agent.db`` is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qa_agent import QARecordStore


@pytest.fixture
def store(tmp_path: Path) -> QARecordStore:
    return QARecordStore(tmp_path / "qa_agent.db")


def test_add_issue_returns_incrementing_id(store):
    first = store.add_issue("a", "Minor", "f.py", 1, "d", "r")
    second = store.add_issue("b", "Critical", "g.py", 2, "d", "r")
    assert first == 1
    assert second == 2


def test_add_issue_defaults(store):
    issue_id = store.add_issue("t", "Major", "f.py", 0, "d", "r")
    row = store.search_issues("*", limit=1)[0]
    assert row["title"] == "t"
    assert row["severity"] == "Major"
    assert row["status"] == "Open"
    assert row["file"] == "f.py"
    assert row["line"] == 0


def test_add_insight_returns_id(store):
    insight_id = store.add_insight("My insight", "use a dict")
    assert insight_id == 1


def test_search_issues_matches_title(store):
    store.add_issue("high severity bug", "Critical", "src.py", 10, "desc", "fix")
    store.add_issue("low severity bug", "Minor", "src.py", 20, "desc", "fix")

    rows = store.search_issues("high", limit=10)
    assert len(rows) == 1
    assert rows[0]["title"] == "high severity bug"
    assert rows[0]["severity"] == "Critical"


def test_search_issues_matches_severity(store):
    store.add_issue("issue", "Critical", "src.py", 1, "desc", "fix")
    rows = store.search_issues("critical", limit=10)
    assert len(rows) == 1


def test_search_issues_matches_description(store):
    store.add_issue("issue", "Minor", "src.py", 1, "the quick brown fox", "fix")
    rows = store.search_issues("quick", limit=10)
    assert len(rows) == 1


def test_search_issues_matches_file(store):
    store.add_issue("issue", "Minor", "src.py", 1, "desc", "fix")
    rows = store.search_issues("src", limit=10)
    assert len(rows) == 1


def test_search_issues_matches_recommendation(store):
    store.add_issue("issue", "Minor", "src.py", 1, "desc", "do the thing")
    rows = store.search_issues("thing", limit=10)
    assert len(rows) == 1


def test_search_issues_respects_limit(store):
    for i in range(5):
        store.add_issue(f"issue {i}", "Minor", "src.py", i, "desc", "fix")
    rows = store.search_issues("*", limit=2)
    assert len(rows) == 2


def test_search_insights(store):
    store.add_insight("first", "alpha content")
    store.add_insight("second", "beta content")

    rows = store.search_insights("beta", limit=10)
    assert len(rows) == 1
    assert rows[0]["content"] == "beta content"


def test_search_insights_by_title(store):
    store.add_insight("my insight", "content")
    rows = store.search_insights("my insight", limit=10)
    assert len(rows) == 1
    assert rows[0]["title"] == "my insight"


def test_rows_are_dicts_with_expected_keys(store):
    store.add_issue("t", "Minor", "f.py", 1, "d", "r")
    row = store.search_issues("*", limit=1)[0]
    assert set(row) == {
        "id", "title", "severity", "status", "file", "line",
        "description", "recommendation", "created_at",
    }


def test_persistence_across_connections(store):
    store.add_issue("persisted", "Minor", "f.py", 1, "d", "r")
    reopened = QARecordStore(store.db_path)
    rows = reopened.search_issues("persisted", limit=10)
    assert len(rows) == 1
