from agno.workflow import Step
from agno.workflow.types import StepInput, StepOutput
import sys
from pathlib import Path

DEEP_RESEARCHER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP_RESEARCHER_ROOT))
from deep_research_agents import market_analyst  # noqa: E402

SYMBOL = "IBM"

def test_market_assessment_step():
    """
    Test the analysis report.
    """

    market_assessment_step = Step(name="Market Assessment", 
                    agent=market_analyst, 
                    description="Market assessment for the stock",
                    )
    query =  f"Summarize recent market developments and the current {SYMBOL} stock price."
    input = StepInput(input=query)    
    result = market_assessment_step.execute(step_input=input)
    print(result)
    assert result is not None
    assert isinstance(result, StepOutput)
    assert result.content is not None