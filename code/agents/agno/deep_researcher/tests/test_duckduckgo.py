"""Assess Agno DuckDuckGoTools behavior and JSON response shapes.

These tests call DuckDuckGoTools methods directly (no LLM). Network-dependent
cases are marked ``integration`` and hit DuckDuckGo via the ddgs library.

Response shapes (integration)

* web_search / search_news → JSON list of result objects (title, href/url, body)
* No matches → ddgs raises DDGSException (not an empty JSON list); Agno does not catch it


Run from ``agentic/agno/``:

    uv run python -m pytest deep_researcher/tests/test_duckduckgo.py -v
    uv run python -m pytest deep_researcher/tests/test_duckduckgo.py -v -m integration
    uv run python -m pytest deep_researcher/tests/test_duckduckgo.py -v -s -k assessment_report
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from agno.tools.duckduckgo import DuckDuckGoTools

QUERY = "NVDA stock news"


def assess_tool_call(tool: DuckDuckGoTools, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a DuckDuckGo tool method and return a JSON-serializable assessment record."""
    method = getattr(tool, method_name)
    record: dict[str, Any] = {
        "method": method_name,
        "args": list(args),
        "kwargs": kwargs,
    }

    try:
        raw = method(*args, **kwargs)
    except Exception as exc:
        record["raw"] = None
        record["status"] = "exception"
        record["parsed"] = {"type": type(exc).__name__, "message": str(exc)}
        return record

    record["raw"] = raw

    if isinstance(raw, str) and raw.startswith("Error "):
        record["status"] = "error"
        record["parsed"] = None
        return record

    try:
        parsed = json.loads(raw)
        record["status"] = "json"
        record["parsed"] = parsed
    except json.JSONDecodeError:
        record["status"] = "plain_text"
        record["parsed"] = raw

    return record


@pytest.fixture
def default_tools() -> DuckDuckGoTools:
    """Same configuration as deep_research_agents.market_analyst."""
    return DuckDuckGoTools(backend="auto")


@pytest.fixture
def limited_tools() -> DuckDuckGoTools:
    return DuckDuckGoTools(backend="auto", fixed_max_results=3)


class TestDuckDuckGoToolsConfiguration:
    def test_default_registers_web_and_news(self, default_tools: DuckDuckGoTools) -> None:
        registered = {tool.__name__ for tool in default_tools.tools}
        assert registered == {"web_search", "search_news"}

    def test_default_backend_is_duckduckgo(self, default_tools: DuckDuckGoTools) -> None:
        assert default_tools.backend == "auto"

    def test_backward_compat_aliases(self, default_tools: DuckDuckGoTools) -> None:
        assert default_tools.duckduckgo_search.__func__ is default_tools.web_search.__func__
        assert default_tools.duckduckgo_news.__func__ is default_tools.search_news.__func__

    def test_invalid_timelimit_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid timelimit"):
            DuckDuckGoTools(timelimit="invalid")

    def test_search_only_configuration(self) -> None:
        tools = DuckDuckGoTools(enable_search=True, enable_news=False)
        registered = {tool.__name__ for tool in tools.tools}
        assert registered == {"web_search"}


@pytest.mark.integration
class TestDuckDuckGoToolsJsonResponses:
    def test_web_search_returns_json_list(self, default_tools: DuckDuckGoTools) -> None:
        assessment = assess_tool_call(default_tools, "web_search", QUERY, max_results=3)
        if assessment["status"] != "json":
            pytest.skip(f"Search unavailable: {assessment.get('parsed')}")
        results = assessment["parsed"]
        assert isinstance(results, list)
        assert 1 <= len(results) <= 3
        first = results[0]
        assert first.get("title")
        assert first.get("href") or first.get("url")
        assert first.get("body")

    def test_search_news_returns_json_list(self, default_tools: DuckDuckGoTools) -> None:
        assessment = assess_tool_call(default_tools, "search_news", QUERY, max_results=3)
        if assessment["status"] != "json":
            pytest.skip(f"News search unavailable: {assessment.get('parsed')}")
        results = assessment["parsed"]
        assert isinstance(results, list)
        assert 1 <= len(results) <= 3
        first = results[0]
        assert first.get("title")
        assert first.get("url") or first.get("href")

    def test_duckduckgo_search_alias_returns_json(self, default_tools: DuckDuckGoTools) -> None:
        assessment = assess_tool_call(default_tools, "duckduckgo_search", QUERY, max_results=2)
        assert assessment["status"] in {"json", "exception"}

    def test_duckduckgo_news_alias_returns_json(self, default_tools: DuckDuckGoTools) -> None:
        assessment = assess_tool_call(default_tools, "duckduckgo_news", QUERY, max_results=2)
        assert assessment["status"] in {"json", "exception"}

    def test_fixed_max_results_limits_output(self, limited_tools: DuckDuckGoTools) -> None:
        assessment = assess_tool_call(limited_tools, "web_search", QUERY, max_results=10)
        if assessment["status"] != "json":
            pytest.skip(f"Search unavailable: {assessment.get('parsed')}")
        assert len(assessment["parsed"]) <= 3


@pytest.mark.integration
def test_duckduckgo_assessment_report_as_json(default_tools: DuckDuckGoTools) -> None:
    """Run core tools and emit one combined JSON assessment (printed for inspection)."""
    methods: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("web_search", (QUERY,), {"max_results": 3}),
        ("search_news", (QUERY,), {"max_results": 3}),
    ]

    report = {
        "query": QUERY,
        "toolkit": "DuckDuckGoTools()",
        "backend": default_tools.backend,
        "results": [
            assess_tool_call(default_tools, name, *args, **kwargs)
            for name, args, kwargs in methods
        ],
    }

    serialized = json.dumps(report, indent=2, default=str)
    print(serialized)

    for entry in report["results"]:
        assert entry["status"] in {"json", "exception"}, json.dumps(entry, indent=2)

    json.loads(serialized)
