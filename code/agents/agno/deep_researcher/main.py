"""
Based on Agno seek demo, create a deep researcher agent.
Read a research paper to extract main information and insights
then build a learning path to learn more.

"""

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.websearch import WebSearchTools
from rich.console import Console
from rich.prompt import Prompt
from agno.media import File

from agno.workflow import Parallel, Step, Workflow

from .deep_research_agents import (
 market_analyst, 
 financial_analyst, 
 technical_analyst, 
 risk_officer, 
 memo_writer, 
 committee_chair
)
"""
Implement the research as a sequential worklow
"""

investment_workflow = Workflow(
    id="investment-workflow",
    name="Investment Review Pipeline",
    steps=[
        Step(name="Market Assessment", agent=market_analyst),
        Parallel(
            Step(name="Fundamental Analysis", agent=financial_analyst),
            Step(name="Technical Analysis", agent=technical_analyst),
            name="Deep Dive",
        ),
        Step(name="Risk Assessment", agent=risk_officer),
        Step(name="Investment Memo", agent=memo_writer),
        Step(name="Committee Decision", agent=committee_chair),
    ],
)

result = investment_workflow.run("Run a full investment review on IBM")
# result.content holds the Committee Decision. Same steps, every run.
print(result.content)