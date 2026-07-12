"""
Agno agent backed by an MLX LLM via an OpenAI-compatible server.

Start the MLX server first (uv run mlx_lm.server --model  mistralai/Mistral-7B-Instruct-v0.3  --port 1337), then run
this agent. Default base URL is http://127.0.0.1:1337/v1.

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

import os

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools import tool
from agno.db.sqlite import SqliteDb
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from pydantic import BaseModel, Field
from typing import List, Optional

# Default URL (e.g. mlx-llm-server or OpenAI-compatible proxy on 1337)
DEFAULT_MLX_BASE_URL = "http://127.0.0.1:1337/v1"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-8B-8bit"
DEFAULT_MLX_TEMPERATURE = 0.4


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
- Recommendation must be one of: Strong Buy, Buy, Hold, Sell, Strong Sell\
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

@tool
def get_word_length(word: str) -> int:
    """Return the length of a word. Use for testing tool use."""
    return len(word)



model = OpenAILike(
    id=DEFAULT_MLX_MODEL,
    base_url=DEFAULT_MLX_BASE_URL,
    temperature=DEFAULT_MLX_TEMPERATURE,
    api_key="no-key",  # mlx-llm-server often accepts any key
)
agent_db = SqliteDb(db_file="tmp/agents.db")
finance_agent = Agent(
    name="Finance Agent",
    model=model,
    instructions=instructions,
    tools=[YFinanceTools(all=True)],
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

