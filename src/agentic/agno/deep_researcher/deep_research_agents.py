
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai.like import OpenAILike
from agno.agent import Agent
from textwrap import dedent
from agno.tools.yfinance import YFinanceTools
import os

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:7999/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-27B-4bit")
DEFAULT_LLM_TEMPERATURE = 0.4
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "localkey")

TEAM_CONTEXT = """
"""

# knowledge managfement
model =OpenAILike(
    id=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_LLM_BASE_URL,
    temperature=DEFAULT_LLM_TEMPERATURE,
    api_key=DEFAULT_LLM_API_KEY,  
)

# DuckDuckGoTools is a tool that can search the web for information. By default it search web and news
# YFinanceTools is a tool that can get the stock price and analyst recommendations for a given stock.
market_analyst = Agent(
    model=model,
    name="Market Analyst",
    tools=[DuckDuckGoTools(backend="auto"), YFinanceTools()],
    instructions=dedent("""
    You are a seasoned Market Analyst.
    1. Summarize recent market developments.
    2. Provide current stock prices and key metrics.
    3. Analyze recent news and market sentiment.
    """),
    user_id="jerome",
    markdown=True
)

financial_analyst = Agent(
    model=model,
    name="Financial Analyst",
    tools=[YFinanceTools()],
    instructions=dedent("""
    You are a seasoned Financial Analyst.
    """),
    user_id="jerome",
    markdown=True
)

technical_analyst = Agent(
    model=model,
    name="Technical Analyst",
    tools=[YFinanceTools()],
    instructions=dedent("""
    You are a seasoned Technical Analyst.
    """),
    user_id="jerome",
    markdown=True
)

risk_officer = Agent(
    model=model,
    name="Risk Officer",
    tools=[YFinanceTools()],
    instructions=dedent("""
    You are a seasoned Risk Officer.
    """),
    user_id="jerome",
    markdown=True
)

memo_writer = Agent(
    model=model,
    name="Memo Writer",
    tools=[YFinanceTools()],
    instructions=dedent("""
    You are a seasoned Memo Writer.
    """),
    user_id="jerome",
    markdown=True
)

committee_chair = Agent(
    model=model,
    name="Committee Chair",
    tools=[YFinanceTools()],
    instructions=dedent("""
    You are a seasoned Committee Chair.
    """),
    user_id="jerome",
    markdown=True
)
