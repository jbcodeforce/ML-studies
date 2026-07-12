"""Integration tests for deep_researcher agents.

Each test runs one agent against the local OpenAI-compatible LLM configured in
``deep_research_agents.py``. Tests skip when ``LLM_BASE_URL`` is unreachable.

Run from ``agentic/agno/``:

    uv run python -m pytest deep_researcher/tests/test_agents.py -v
    uv run python -m pytest deep_researcher/tests/test_agents.py -v -k market_analyst
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from agno.agent import Agent

DEEP_RESEARCHER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP_RESEARCHER_ROOT))

from deep_research_agents import (  # noqa: E402
    committee_chair,
    financial_analyst,
    market_analyst,
    memo_writer,
    risk_officer,
    technical_analyst,
)

SYMBOL = "IBM"


def llm_available() -> bool:
    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:7999/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "localkey")
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


@pytest.fixture(scope="module")
def require_llm() -> None:
    if not llm_available():
        pytest.skip("Local OpenAI-compatible LLM not reachable (LLM_BASE_URL)")


def run_agent(agent: Agent, prompt: str) -> dict[str, Any]:
    response = agent.run(prompt)
    assert response.content is not None
    assert response.content.strip()

    record: dict[str, Any] = {
        "agent": agent.name,
        "prompt": prompt,
        "content_preview": response.content.strip()[:500],
    }
    if response.tools:
        record["tools_used"] = [tool.tool_name for tool in response.tools if tool.tool_name]
    return record


@pytest.mark.integration
def test_market_analyst(require_llm: None) -> None:
    record = run_agent(
        market_analyst,
        f"Summarize recent market developments and the current {SYMBOL} stock price.",
    )
    print(json.dumps(record, indent=2))
    assert record["agent"] == "Market Analyst"


@pytest.mark.integration
def test_financial_analyst(require_llm: None) -> None:
    record = run_agent(
        financial_analyst,
        f"Provide a brief fundamental analysis for {SYMBOL}.",
    )
    print(json.dumps(record, indent=2))
    assert record["agent"] == "Financial Analyst"


@pytest.mark.integration
def test_technical_analyst(require_llm: None) -> None:
    record = run_agent(
        technical_analyst,
        f"Provide a brief technical analysis outlook for {SYMBOL} stock.",
    )
    print(json.dumps(record, indent=2))
    assert record["agent"] == "Technical Analyst"


@pytest.mark.integration
def test_risk_officer(require_llm: None) -> None:
    record = run_agent(
        risk_officer,
        f"List the main investment risks for {SYMBOL}.",
    )
    print(json.dumps(record, indent=2))
    assert record["agent"] == "Risk Officer"


@pytest.mark.integration
def test_memo_writer(require_llm: None) -> None:
    record = run_agent(
        memo_writer,
        f"Draft a short investment memo outline for {SYMBOL}.",
    )
    print(json.dumps(record, indent=2))
    assert record["agent"] == "Memo Writer"


@pytest.mark.integration
def test_committee_chair(require_llm: None) -> None:
    record = run_agent(
        committee_chair,
        f"Provide a committee decision summary for a potential {SYMBOL} investment.",
    )
    print(json.dumps(record, indent=2))
    assert record["agent"] == "Committee Chair"
