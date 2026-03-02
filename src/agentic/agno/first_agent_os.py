"""
Agent OS to expose a FASTapi that could be connected from https://os.agno.com/
Running with ollama server.


Example prompts to try:
- "What's the current price of AAPL?"
- "Compare NVDA and AMD — which looks stronger?"
- "Give me a quick investment brief on Microsoft"
- "What's Tesla's P/E ratio and how does it compare to the industry?"
- "Show me the key metrics for the FAANG stocks"
"""

from pathlib import Path
from agno.os import AgentOS
from first_mlx_agent_with_tool import finance_agent


# ---------------------------------------------------------------------------
# AgentOS Config
# ---------------------------------------------------------------------------
config_path = str(Path(__file__).parent.joinpath("config.yaml"))

agent_os = AgentOS(
    id="Quick Start AgentOS",
    agents=[
        finance_agent,
    ],
    config=config_path,
    tracing=True
)
app = agent_os.get_app()

# ---------------------------------------------------------------------------
# Run AgentOS
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent_os.serve(app="first_agent_os:app", reload=True)

