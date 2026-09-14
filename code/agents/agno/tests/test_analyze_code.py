"""Unit tests for the heuristic static analyzer ``qa_agent._analyze_code``.

These exercise the pure function directly, so no LLM server is required.
"""

from __future__ import annotations

import json

import pytest

from qa_agent import _analyze_code


def test_missing_file_returns_error_dict():
    result = _analyze_code("/does/not/exist.py")
    assert result["error"] == "File not found: /does/not/exist.py"
    assert result["total_lines"] == 0
    assert result["functions"] == 0
    assert result["smells"] == []


def test_missing_file_result_is_json_serializable():
    result = _analyze_code("/does/not/exist.py")
    json.dumps(result)  # must not raise


def test_clean_file_has_no_smells(sample_files):
    result = _analyze_code(str(sample_files / "good.py"))
    assert result["error"] is None
    assert result["smells"] == []
    assert result["functions"] == 2  # greet, main
    assert result["total_lines"] == 5


def test_counts_functions(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    # add, run, plus do_work() referenced but not defined -> def count is 2
    assert result["functions"] == 2


def test_detects_hardcoded_secret(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    secrets = [s for s in result["smells"] if "secret" in s.lower()]
    assert secrets, f"expected a hardcoded-secret finding, got {result['smells']}"


def test_detects_eval(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    smells = " ".join(result["smells"]).lower()
    assert "eval" in smells


def test_detects_bare_except(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    smells = " ".join(result["smells"]).lower()
    assert "bare" in smells and "except" in smells


def test_detects_sql_injection(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    smells = " ".join(result["smells"]).lower()
    assert "sql injection" in smells


def test_detects_none_comparison(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    smells = " ".join(result["smells"]).lower()
    assert "is" in smells and "none" in smells


def test_detects_todo_marker(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    smells = " ".join(result["smells"]).lower()
    assert "todo" in smells


def test_respects_max_lines(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"), max_lines=3)
    assert result["lines_scanned"] <= 3
    smells = " ".join(result["smells"]).lower()
    assert "exceeds" in smells


def test_report_shape(sample_files):
    result = _analyze_code(str(sample_files / "good.py"))
    assert set(result) == {
        "file", "total_lines", "functions", "lines_scanned", "smells",
    }
    assert result["file"] == str(sample_files / "good.py")
