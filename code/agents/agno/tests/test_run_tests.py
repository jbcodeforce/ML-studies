"""Unit tests for ``qa_agent._run_tests``.

The command is executed in a throwaway temp directory so we never run the
real project test-suite (and never depend on a network).
"""

from __future__ import annotations

import os

import pytest

from qa_agent import _run_tests


def test_invalid_timeout():
    result = _run_tests("pytest -q", timeout=0)
    assert result["status"] == "invalid"
    assert "timeout" in result["error"].lower()


def test_empty_command():
    result = _run_tests("   ")
    assert result["status"] == "invalid"
    assert "command" in result["error"].lower()


def test_success_parses_pass_count(workspace, monkeypatch):
    script = workspace / "ok.sh"
    script.write_text("echo '3 passed, 0 failed in 0.01s'")
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))

    result = _run_tests(f"bash {script}", timeout=10)
    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert result["tests_passed"] == 3
    assert result["tests_failed"] is None
    assert "passed" in result["summary"].lower()


def test_failure_reports_nonzero_exit(workspace, monkeypatch):
    script = workspace / "fail.sh"
    script.write_text("echo '2 passed, 1 failed' ; exit 1")
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))

    result = _run_tests(f"bash {script}", timeout=10)
    assert result["status"] == "completed"
    assert result["exit_code"] == 1
    assert result["tests_passed"] == 2
    assert result["tests_failed"] == 1
    assert "failures" in result["summary"].lower()


def test_captures_stderr(workspace, monkeypatch):
    script = workspace / "err.sh"
    script.write_text("echo 'boom' 1>&2 ; exit 2")
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))

    result = _run_tests(f"bash {script}", timeout=10)
    tail = " ".join(result["tail"]).lower()
    assert "boom" in tail


def test_tail_is_bounded(workspace, monkeypatch):
    script = workspace / "big.sh"
    script.write_text("\n".join(f"line {i}" for i in range(200)))
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))

    result = _run_tests(f"bash {script}", timeout=10)
    # tail holds the last ~80 lines, not all 200
    assert 0 < len(result["tail"]) <= 80
