"""
Agno agent backed by an Ollama with tool calling. 

Start the Ollama server first ollama serve

Agent demonstrates:
* use of system prompt via instructions variable
* use of tools
* use of SqliteDb to store context
* use of BaseModel to define the structured output
* use of stream=True to print the response in real time
* use of add_datetime_to_context=True to add the datetime to the context
* use of markdown=True to print the response in markdown format

Example prompts to try:
- "What's the current price of AAPL?"
- "Compare NVDA and AMD — which looks stronger?"
- "Give me a quick investment brief on Microsoft"
- "What's Tesla's P/E ratio and how does it compare to the industry?"
- "Show me the key metrics for the FAANG stocks"
"""

import json
from datetime import datetime, timezone
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools import tool
from agno.db.sqlite import SqliteDb

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools


from pydantic import BaseModel, Field
from typing import List, Optional

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "mistral:7b-instruct"
DEFAULT_LLM_TEMPERATURE = 0.4


instructions = """\
You are a Finance Agent — a data-driven analyst who retrieves market data,
computes key ratios, and produces concise, decision-ready insights.

## Workflow

1. Retrieve
   - Fetch: price, change %, market cap, P/E, EPS, 52-week range
   - For comparisons, pull the same fields for each ticker

2. Analyze
   - Compute ratios (P/E, P/S, margins) when not already provided
   - Key drivers and risks — 2-3 bullets max
   - Facts only, no speculation

3. Recommend
   - Based on the data, provide a clear recommendation
   - Be decisive but note this is not personalized advice

## Rules

- Source: Yahoo Finance. Always note the timestamp.
- Missing data? Say "N/A" and move on.
- - Recommendation must be one of: Strong Buy, Buy, Hold, Sell, Strong Sell\
"""

class StockAnalysis(BaseModel):
    """Structured output for stock analysis."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., NVDA)")
    company_name: str = Field(..., description="Full company name")
    current_price: float = Field(..., description="Current stock price in USD")
    market_cap: str = Field(..., description="Market cap (e.g., '3.2T' or '150B')")
    pe_ratio: Optional[float] = Field(None, description="P/E ratio, if available")
    week_52_high: float = Field(..., description="52-week high price")
    week_52_low: float = Field(..., description="52-week low price")
    summary: str = Field(..., description="One-line summary of the stock")
    key_drivers: List[str] = Field(..., description="2-3 key growth drivers")
    key_risks: List[str] = Field(..., description="2-3 key risks")
    recommendation: str = Field(
        ..., description="One of: Strong Buy, Buy, Hold, Sell, Strong Sell"
    )


model = Ollama(
    id=DEFAULT_LLM_MODEL
)
agent_db = SqliteDb(db_file="tmp/agents.db")


finance_agent = Agent(
    name="Finance Agent",
    model=model,
    instructions=instructions,
    tools=[YFinanceTools(all=True), DuckDuckGoTools(fixed_max_results=2)],
    db=agent_db,
    output_schema=StockAnalysis,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
)


if __name__ == "__main__":
    print("Chat with mistral until entering an empty question")
    done = False
    agent = finance_agent
    while not done:
        print("Question >:")
        question = input()
        if not question or 'bye' in question:
            done = True
        else:
            response = agent.run(question)
            analysis: StockAnalysis = response.content
            print(analysis)
            print("\n\n")

