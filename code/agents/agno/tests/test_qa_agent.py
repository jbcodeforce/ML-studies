"""Unit tests for every function in ``qa_agent``.

The module exposes a small set of helpers, two core logic functions, six
``@tool`` wrappers, a persistence layer, and a CLI.  These tests exercise the
pure/deterministic parts directly, so no LLM server is required.

Coverage targets (function -> representative test):

* helpers            -> ``test_ts_format``, ``test_iso_format``
* ``QARecordStore``  -> ``test_record_store_*``
* ``_analyze_code``  -> ``test_analyze_code_*``
* ``_run_tests``     -> ``test_run_tests_*``
* ``@tool`` wrappers  -> ``test_tool_*``
* ``build_knowledge_base`` -> ``test_build_knowledge_base_*``
* ``_instructions``  -> ``test_instructions_*``
* ``build_agent``    -> ``test_build_agent_*``
* ``verify_model_available`` -> ``test_verify_model_available_*``
* ``_print_header``  -> ``test_print_header_*``
* ``_run_once``      -> ``test_run_once_*``
* ``parse_args``     -> ``test_parse_args_*``
* ``main``           -> ``test_main_*``
"""

from __future__ import annotations

import io
import json
import pytest
from pathlib import Path
import qa_agent
from qa_agent import (
    QAReport,
    Issue,
    QARecordStore,
    _analyze_code,
    _run_tests,
    _ts,
    _iso,
    analyze_code,
    build_agent,
    build_knowledge_base,
    document_issue,
    main,
    parse_args,
    record_store,
    run_tests,
    save_insight,
    search_insights,
    search_issues,
    verify_model_available,
)


def test_record_store_add_insight_returns_id(store):
    assert store.add_insight("My insight", "use a dict") == 1


def test_record_store_search_issues_matches_title(store):
    store.add_issue("high severity bug", "Critical", "src.py", 10, "desc", "fix")
    store.add_issue("low severity bug", "Minor", "src.py", 20, "desc", "fix")
    rows = store.search_issues("high", limit=10)
    assert len(rows) == 1
    assert rows[0]["title"] == "high severity bug"
    assert rows[0]["severity"] == "Critical"


def test_record_store_search_issues_matches_severity(store):
    store.add_issue("issue", "Critical", "src.py", 1, "desc", "fix")
    rows = store.search_issues("critical", limit=10)
    assert len(rows) == 1


def test_record_store_search_issues_matches_description(store):
    store.add_issue("issue", "Minor", "src.py", 1, "the quick brown fox", "fix")
    rows = store.search_issues("quick", limit=10)
    assert len(rows) == 1


def test_record_store_search_issues_matches_file(store):
    store.add_issue("issue", "Minor", "src.py", 1, "desc", "fix")
    rows = store.search_issues("src", limit=10)
    assert len(rows) == 1


def test_record_store_search_issues_matches_recommendation(store):
    store.add_issue("issue", "Minor", "src.py", 1, "desc", "do the thing")
    rows = store.search_issues("thing", limit=10)
    assert len(rows) == 1


def test_record_store_search_issues_respects_limit(store):
    for i in range(5):
        store.add_issue(f"issue {i}", "Minor", "src.py", i, "desc", "fix")
    rows = store.search_issues("issue ", limit=2)
    assert len(rows) == 2


def test_record_store_search_insights(store):
    store.add_insight("first", "alpha content")
    store.add_insight("second", "beta content")
    rows = store.search_insights("beta", limit=10)
    assert len(rows) == 1
    assert rows[0]["content"] == "beta content"


def test_record_store_search_insights_by_title(store):
    store.add_insight("my insight", "content")
    rows = store.search_insights("my insight", limit=10)
    assert len(rows) == 1
    assert rows[0]["title"] == "my insight"


def test_record_store_rows_are_dicts_with_expected_keys(store):
    store.add_issue("t", "Minor", "f.py", 1, "d", "r")
    row = store.search_issues("t", limit=1)[0]
    assert set(row) == {
        "id", "title", "severity", "status", "file", "line",
        "description", "recommendation", "created_at",
    }


def test_record_store_persistence_across_connections(store):
    store.add_issue("persisted", "Minor", "f.py", 1, "d", "r")
    reopened = QARecordStore(store.db_path)
    rows = reopened.search_issues("persisted", limit=10)
    assert len(rows) == 1



# ---------------------------------------------------------------------------
# _analyze_code
# ---------------------------------------------------------------------------
def test_analyze_code_missing_file_returns_error_dict():
    result = _analyze_code("/does/not/exist.py")
    assert result["error"] == "File not found: /does/not/exist.py"
    assert result["total_lines"] == 0
    assert result["functions"] == 0
    assert result["smells"] == []


def test_analyze_code_missing_file_is_json_serializable():
    result = _analyze_code("/does/not/exist.py")
    json.dumps(result)  # must not raise


def test_analyze_code_clean_file_has_no_smells(sample_files):
    result = _analyze_code(str(sample_files / "good.py"))
    print(result)
    assert result["smells"] == []
    assert result["functions"] == 2  # greet, main
    assert result["total_lines"] == 7


def test_analyze_code_counts_functions(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    # add, run, plus do_work() referenced but not defined -> def count is 2
    assert result["functions"] == 2


def test_analyze_code_detects_hardcoded_secret(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    secrets = [s for s in result["smells"] if "secret" in s.lower()]
    assert secrets, f"expected a hardcoded-secret finding, got {result['smells']}"


def test_analyze_code_detects_eval(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    smells = " ".join(result["smells"]).lower()
    assert "eval" in smells


def test_analyze_code_detects_bare_except(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    smells = " ".join(result["smells"]).lower()
    assert "bare" in smells and "except" in smells


def test_analyze_code_detects_sql_injection(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"))
    smells = " ".join(result["smells"]).lower()
    print(smells)
    assert "hardcoded secret" in smells
    assert "sql injection" in smells
    assert "is dangerous" in smells
    assert "todo marker" in smells


def test_analyze_code_respects_max_lines(sample_files):
    result = _analyze_code(str(sample_files / "bad.py"), max_lines=3)
    assert result["lines_scanned"] <= 3
    smells = " ".join(result["smells"]).lower()
    assert "exceeds" in smells


def test_analyze_code_report_shape(sample_files):
    result = _analyze_code(str(sample_files / "good.py"))
    assert set(result) == {
        "file", "total_lines", "functions", "lines_scanned", "smells",
    }
    assert result["file"] == str(sample_files / "good.py")


# ---------------------------------------------------------------------------
# _run_tests
# ---------------------------------------------------------------------------
def test_run_tests_invalid_timeout():
    result = _run_tests("pytest -q", timeout=0)
    assert result["status"] == "invalid"
    assert "timeout" in result["error"].lower()


def test_run_tests_empty_command():
    result = _run_tests("   ")
    assert result["status"] == "invalid"
    assert "command" in result["error"].lower()


def test_run_tests_success_parses_pass_count(workspace, monkeypatch):
    script = workspace / "ok.sh"
    script.write_text("echo '3 passed, 0 failed in 0.01s'")
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))

    result = _run_tests(f"bash {script}", timeout=10)
    print(result)
    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert result["tests_passed"] == 3
    assert result["tests_failed"] == 0
    assert "passed" in result["summary"].lower()


def test_run_tests_failure_reports_nonzero_exit(workspace, monkeypatch):
    script = workspace / "fail.sh"
    script.write_text("echo '2 passed, 1 failed' ; exit 1")
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))

    result = _run_tests(f"bash {script}", timeout=10)
    assert result["status"] == "completed"
    assert result["exit_code"] == 1
    assert result["tests_passed"] == 2
    assert result["tests_failed"] == 1
    assert "failures" in result["summary"].lower()


def test_run_tests_captures_stderr(workspace, monkeypatch):
    script = workspace / "err.sh"
    script.write_text("echo 'boom' 1>&2 ; exit 2")
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))

    result = _run_tests(f"bash {script}", timeout=10)
    print(result)
    tail = " ".join(result["tail"]).lower()
    assert "boom" in tail


def test_run_tests_tail_is_bounded(workspace, monkeypatch):
    script = workspace / "big.sh"
    script.write_text("\n".join(f"line {i}" for i in range(200)))
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))

    result = _run_tests(f"bash {script}", timeout=10)
    # tail holds the last ~80 lines, not all 200
    assert 0 < len(result["tail"]) <= 80


# ---------------------------------------------------------------------------
# @tool wrappers

def test_tool_search_insight_wrap(isolated_records):
    save_insight(title="learned", content="use dicts")
    payload = search_insight(query="use", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1
    assert data["insights"][0]["content"] == "use dicts"


def test_tool_analyze_code_returns_json(sample_files):
    payload = analyze_code(str(sample_files / "good.py"))
    data = json.loads(payload)
    assert data["functions"] == 2
    assert data["smells"] == []


def test_tool_run_tests_defaults_to_pytest(workspace, monkeypatch):
    monkeypatch.setenv("QA_AGENT_WORKSPACE", str(workspace))
    payload = run_tests("", timeout=5)  # empty command -> invalid
    data = json.loads(payload)
    assert data["status"] == "invalid"


def test_tool_document_issue_returns_id(isolated_records):
    payload = document_issue(
        title="A bug", severity="Critical", status="Open",
        file="src.py", line=42, description="boom", recommendation="fix",
    )
    data = json.loads(payload)
    assert data["status"] == "documented"
    assert data["issue_id"] == 1
    assert data["title"] == "A bug"
    assert data["severity"] == "Critical"


def test_tool_document_issue_defaults(isolated_records):
    payload = document_issue(title="t", severity="Minor")
    data = json.loads(payload)
    assert data["issue_id"] == 1
    # defaults in the tool: file -> "N/A", line -> 0, status -> "Open"
    row = isolated_records.search_issues("*", limit=1)[0]
    assert row["file"] == "N/A"
    assert row["line"] == 0
    assert row["status"] == "Open"


def test_tool_document_then_search_issues_roundtrip(isolated_records):
    document_issue(
        title="findable bug", severity="Major", file="src.py", line=7,
        description="desc", recommendation="fix",
    )
    payload = search_issues(query="findable", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1
    assert data["issues"][0]["title"] == "findable bug"


def test_tool_save_and_search_insights_roundtrip(isolated_records):
    save_insight(title="learned", content="use dicts")
    payload = search_insights(query="use", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1
    assert data["insights"][0]["content"] == "use dicts"


def test_tool_search_insights_wrap(isolated_records):
    save_insight(title="learned", content="use dicts")
    payload = search_insight(query="use", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1
    assert data["insights"][0]["title"] == "learned"


def test_tool_search_issues_wrap(isolated_records):
    document_issue(title="findable bug", severity="Major", file="src.py", line=7)
    payload = search_issues(query="findable", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1
    assert data["issues"][0]["severity"] == "Major"


def test_tool_search_insight_wrap(isolated_records):
    save_insight(title="learned", content="use dicts")
    payload = search_insight(query="use", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1
    assert data["insights"][0]["content"] == "use dicts"


def test_tool_search_issues_empty_query(isolated_records):
    document_issue(title="x", severity="Minor", file="f", line=1)
    payload = search_issues(query="", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1


# ---------------------------------------------------------------------------
# build_knowledge_base
# ---------------------------------------------------------------------------
def test_build_knowledge_base_no_embedder(monkeypatch):
    monkeypatch.setattr(qa_agent, "EMBEDDER_MODEL", None)
    assert build_knowledge_base() is None


def test_build_knowledge_base_with_embedder(monkeypatch):
    monkeypatch.setattr(qa_agent, "EMBEDDER_MODEL", "fake-embedder-model")
    monkeypatch.setattr(qa_agent, "EMBEDDER_BASE_URL", "http://localhost:9999/v1")
    kb = build_knowledge_base()
    assert kb is not None


# ---------------------------------------------------------------------------
# _instructions
# ---------------------------------------------------------------------------
def test_instructions_contains_severity_levels():
    instructions = _instructions(use_kb=False)
    for level in ("Critical", "Major", "Minor", "Info"):
        assert level in instructions


def test_instructions_contains_reporting_rules():
    instructions = _instructions(use_kb=False)
    assert "file:line" in instructions.lower()
    assert "reproduction" in instructions.lower()


def test_instructions_with_kb_mentions_knowledge_base():
    instructions = _instructions(use_kb=True)
    assert "knowledge base" in instructions.lower()


def test_instructions_without_kb_mentions_memory_note():
    instructions = _instructions(use_kb=False)
    assert "no semantic knowledge base" in instructions.lower()


# ---------------------------------------------------------------------------
# build_agent
# ---------------------------------------------------------------------------
def test_build_agent_returns_agent_with_tools(tmp_path):
    agent = build_agent(workspace=tmp_path)
    assert agent is not None
    tool_names = {t.tool_name for t in agent.tools}
    assert "analyze_code" in tool_names
    assert "run_tests" in tool_names
    assert "document_issue" in tool_names
    assert "search_issues" in tool_names
    assert "save_insight" in tool_names
    assert "search_insights" in tool_names


def test_build_agent_without_kb(monkeypatch, tmp_path):
    monkeypatch.setattr(qa_agent, "EMBEDDER_MODEL", None)
    agent = build_agent(workspace=tmp_path)
    assert agent is not None


# ---------------------------------------------------------------------------
# verify_model_available
# ---------------------------------------------------------------------------
def test_verify_model_available_reaches_server(monkeypatch):
    import httpx

    class Response(httpx.Response):
        def __init__(self):
            self.status_code = 200
            self._content = json.dumps({"data": [{"id": "qwen"}]}).encode()
            self.request = httpx.Request("GET", "http://localhost/models")

    monkeypatch.setattr(
        qa_agent.httpx, "get",
        lambda url, headers=None, timeout=None: Response(),
    )
    result = verify_model_available(model="qwen", base_url="http://localhost",
                                    api_key="k", verbose=False)
    assert result is True


def test_verify_model_available_unknown_model_warns(monkeypatch, capsys):
    import httpx

    class Response(httpx.Response):
        def __init__(self):
            self.status_code = 200
            self._content = json.dumps({"data": [{"id": "other"}]}).encode()
            self.request = httpx.Request("GET", "http://localhost/models")

    monkeypatch.setattr(
        qa_agent.httpx, "get",
        lambda url, headers=None, timeout=None: Response(),
    )
    result = verify_model_available(model="missing", base_url="http://localhost",
                                    api_key="k", verbose=True)
    assert result is True
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_verify_model_available_server_down(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    import httpx

    monkeypatch.setattr(qa_agent.httpx, "get", boom)
    result = verify_model_available(model="qwen", base_url="http://localhost",
                                    api_key="k", verbose=True)
    assert result is False
    assert "could not reach" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# _print_header
# ---------------------------------------------------------------------------
def test_print_header_prints_table(capsys):
    _print_header()
    out = capsys.readouterr().out
    assert "QA Agent" in out


# ---------------------------------------------------------------------------
# _run_once
# ---------------------------------------------------------------------------
def test_run_once_no_server_returns_1(monkeypatch):
    import httpx

    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(qa_agent.httpx, "get", boom)
    from rich.console import Console

    rc = _run_once("hello", Console(), structured=False, workspace=Path("."),
                   model="m", base_url="http://localhost", api_key="k", verbose=False)
    assert rc == 1


def test_run_once_structured_success(monkeypatch, tmp_path):
    import httpx

    class Response(httpx.Response):
        def __init__(self):
            self.status_code = 200
            self._content = "structured answer"
            self.request = httpx.Request("GET", "http://localhost/models")

    monkeypatch.setattr(
        qa_agent.httpx, "get",
        lambda url, headers=None, timeout=None: Response(),
    )
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_BASE_URL", "http://localhost")
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_MODEL", "m")
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_API_KEY", "k")
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_TEMPERATURE", 0.4)

    from rich.console import Console

    rc = _run_once("q", Console(), structured=True, workspace=tmp_path,
                   model="m", base_url="http://localhost", api_key="k", verbose=False)
    assert rc == 0


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
def test_parse_args_defaults():
    ns = parse_args([])
    assert ns.query is None
    assert not ns.structured
    assert not ns.interactive


def test_parse_args_query_and_structured():
    ns = parse_args(["--query", "hi", "--structured"])
    assert ns.query == "hi"
    assert ns.structured


def test_parse_args_flags():
    ns = parse_args(["-w", ".", "--no-model-check", "--interactive"])
    assert ns.workspace == Path(".")
    assert ns.no_model_check
    assert ns.interactive


def test_parse_args_accepts_none():
    ns = parse_args(None)
    assert ns is not None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def test_main_missing_workspace_returns_2(tmp_path):
    rc = main(["--workspace", str(tmp_path / "does_not_exist")])
    assert rc == 2


def test_main_no_args_interactive(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_BASE_URL", "http://localhost")
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_MODEL", "m")
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_API_KEY", "k")
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_TEMPERATURE", 0.4)

    import qa_agent

    monkeypatch.setattr(qa_agent.httpx, "get",
                        lambda *a, **k: _mock_models_response())

    # Feed 'exit' on stdin to end the REPL.
    monkeypatch.setattr("sys.stdin", io.StringIO("exit\n"))
    rc = main([])
    assert rc == 0


def test_main_structured_query(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_BASE_URL", "http://localhost")
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_MODEL", "m")
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_API_KEY", "k")
    monkeypatch.setattr(qa_agent, "DEFAULT_LLM_TEMPERATURE", 0.4)

    import qa_agent

    monkeypatch.setattr(qa_agent.httpx, "get",
                        lambda *a, **k: _mock_models_response())

    rc = main(["--query", "hi", "--structured", "-w", str(tmp_path),
               "--no-model-check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "structured answer" in out


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
def test_issue_model_valid():
    issue = Issue(title="t", severity="Critical", status="Open",
                  file="f.py", line=3, description="d", recommendation="r")
    assert issue.title == "t"
    assert issue.line == 3


def test_issue_model_defaults():
    issue = Issue(title="t", severity="Minor", status="Open",
                  file="f.py", description="d", recommendation="r")
    assert issue.line == 0
    assert issue.reproduction_steps == ""


def test_qareport_model_valid():
    report = QAReport(project="proj", summary="all good",
                      issues=[Issue(title="t", severity="Minor", status="Open",
                                    file="f", description="d", recommendation="r")])
    assert report.tests_run is None
    assert report.coverage is None


def test_qareport_model_with_counts():
    report = QAReport(project="proj", summary="done", tests_run=10,
                      tests_passed=8, tests_failed=2, coverage=80.0)
    assert report.tests_run == 10
    assert report.tests_failed == 2
    assert report.coverage == 80.0


# ---------------------------------------------------------------------------
# Module-level record store smoke test
# ---------------------------------------------------------------------------
def test_module_record_store_searchable():
    document_issue(title="module issue", severity="Info", file="m.py", line=1)
    payload = search_issues(query="module issue", limit=10)
    data = json.loads(payload)
    assert data["count"] == 1


# ---------------------------------------------------------------------------
# Helpers for tests that stub the /v1/models endpoint
# ---------------------------------------------------------------------------
def _mock_models_response():
    import httpx

    class Response(httpx.Response):
        def __init__(self):
            self.status_code = 200
            self._content = json.dumps({"data": [{"id": "m"}]}).encode()
            self.request = httpx.Request("GET", "http://localhost/models")

    return Response()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path: Path) -> QARecordStore:
    return QARecordStore(tmp_path / "qa_agent.db")


@pytest.fixture(autouse=True)
def _qaagent_path():
    """Expose the qa_agent module path for tests that need it."""
    return Path(qa_agent.__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
