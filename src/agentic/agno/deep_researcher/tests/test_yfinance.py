"""Assess Agno YFinanceTools behavior and JSON response shapes.

These tests call YFinanceTools methods directly (no LLM). Network-dependent
cases are marked ``integration`` and hit Yahoo Finance via yfinance.

Run from ``src/``:

    uv run pytest agentic/agno/deep_researcher/tests/test_yfinance.py -v
    uv run pytest agentic/agno/deep_researcher/tests/test_yfinance.py -v -m integration
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from agno.tools.yfinance import YFinanceTools

SYMBOL = "IBM"
PRICE_PATTERN = re.compile(r"^\d+\.\d{4}$")


def assess_tool_call(tool: YFinanceTools, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a YFinance tool method and return a JSON-serializable assessment record."""
    method = getattr(tool, method_name)
    raw = method(*args, **kwargs)
    record: dict[str, Any] = {
        "method": method_name,
        "args": list(args),
        "kwargs": kwargs,
        "raw": raw,
    }

    if raw.startswith("Error ") or raw.startswith("Could not fetch"):
        record["status"] = "error"
        record["parsed"] = None
        return record

    if method_name == "get_current_stock_price" and PRICE_PATTERN.match(raw):
        record["status"] = "price_string"
        record["parsed"] = {"price": float(raw)}
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
def default_tools() -> YFinanceTools:
    """Same configuration as deep_research_agents.market_analyst."""
    return YFinanceTools()


@pytest.fixture
def all_tools() -> YFinanceTools:
    return YFinanceTools(all=True)


class TestYFinanceToolsConfiguration:
    def test_default_registers_only_stock_price(self, default_tools: YFinanceTools) -> None:
        registered = {tool.__name__ for tool in default_tools.tools}
        assert registered == {"get_current_stock_price"}

    def test_all_flag_registers_every_method(self, all_tools: YFinanceTools) -> None:
        registered = {tool.__name__ for tool in all_tools.tools}
        assert registered == {
            "get_current_stock_price",
            "get_company_info",
            "get_stock_fundamentals",
            "get_income_statements",
            "get_key_financial_ratios",
            "get_analyst_recommendations",
            "get_company_news",
            "get_technical_indicators",
            "get_historical_stock_prices",
        }


@pytest.mark.integration
class TestYFinanceToolsJsonResponses:
    def test_get_current_stock_price_returns_decimal_string(self, all_tools: YFinanceTools) -> None:
        assessment = assess_tool_call(all_tools, "get_current_stock_price", SYMBOL)
        assert assessment["status"] == "price_string"
        assert assessment["parsed"]["price"] > 0

    def test_get_company_info_returns_json_profile(self, all_tools: YFinanceTools) -> None:
        assessment = assess_tool_call(all_tools, "get_company_info", SYMBOL)
        assert assessment["status"] == "json"
        profile = assessment["parsed"]
        assert profile["Symbol"] == SYMBOL
        assert profile["Name"]
        assert "Current Stock Price" in profile
        assert profile["Sector"]

    def test_get_stock_fundamentals_returns_json(self, all_tools: YFinanceTools) -> None:
        assessment = assess_tool_call(all_tools, "get_stock_fundamentals", SYMBOL)
        assert assessment["status"] == "json"
        fundamentals = assessment["parsed"]
        assert fundamentals["symbol"] == SYMBOL
        assert fundamentals["company_name"]
        assert "market_cap" in fundamentals
        assert "pe_ratio" in fundamentals

    def test_get_analyst_recommendations_returns_json(self, all_tools: YFinanceTools) -> None:
        assessment = assess_tool_call(all_tools, "get_analyst_recommendations", SYMBOL)
        assert assessment["status"] == "json"
        assert isinstance(assessment["parsed"], dict)

    def test_get_historical_stock_prices_returns_json(self, all_tools: YFinanceTools) -> None:
        assessment = assess_tool_call(
            all_tools,
            "get_historical_stock_prices",
            SYMBOL,
            period="5d",
            interval="1d",
        )
        assert assessment["status"] == "json"
        prices = assessment["parsed"]
        assert isinstance(prices, dict)
        assert len(prices) >= 1
        first_row = next(iter(prices.values()))
        assert "Close" in first_row

    def test_get_company_news_returns_json_list(self, all_tools: YFinanceTools) -> None:
        assessment = assess_tool_call(all_tools, "get_company_news", SYMBOL, num_stories=2)
        assert assessment["status"] == "json"
        news = assessment["parsed"]
        assert isinstance(news, list)
        assert len(news) <= 2

    def test_get_technical_indicators_returns_json(self, all_tools: YFinanceTools) -> None:
        assessment = assess_tool_call(all_tools, "get_technical_indicators", SYMBOL, period="5d")
        assert assessment["status"] == "json"
        indicators = assessment["parsed"]
        assert isinstance(indicators, dict)
        assert len(indicators) >= 1

    def test_invalid_symbol_returns_error_string(self, all_tools: YFinanceTools) -> None:
        assessment = assess_tool_call(all_tools, "get_current_stock_price", "NOTAREALSYMBOL999")
        assert assessment["status"] in {"error", "plain_text", "price_string"}


@pytest.mark.integration
def test_yfinance_assessment_report_as_json(all_tools: YFinanceTools) -> None:
    """Run a subset of tools and emit one combined JSON assessment (printed for inspection)."""
    methods: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("get_current_stock_price", (SYMBOL,), {}),
        ("get_company_info", (SYMBOL,), {}),
        ("get_stock_fundamentals", (SYMBOL,), {}),
        ("get_analyst_recommendations", (SYMBOL,), {}),
        ("get_historical_stock_prices", (SYMBOL,), {"period": "5d", "interval": "1d"}),
    ]

    report = {
        "symbol": SYMBOL,
        "toolkit": "YFinanceTools(all=True)",
        "results": [
            assess_tool_call(all_tools, name, *args, **kwargs)
            for name, args, kwargs in methods
        ],
    }

    serialized = json.dumps(report, indent=2, default=str)
    print(serialized)

    for entry in report["results"]:
        assert entry["status"] != "error", json.dumps(entry, indent=2)

    #json.loads(serialized)
