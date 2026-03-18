"""
Agno agent backed by an Ollama with tool calling. Implement the self learning 
agent pattern that can save insights to a knowledge base.
It uses human in the loop to approve or reject the learnings.

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
from agno.utils import pprint
from agno.db.sqlite import SqliteDb

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

from agno.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.reader.text_reader import TextReader
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType
from rich.console import Console
from rich.prompt import Prompt

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "mistral:7b-instruct"
DEFAULT_LLM_TEMPERATURE = 0.4


instructions = """\
You are a Finance Agent that learns and improves over time.

You have two special abilities:
1. Search your knowledge base for previously saved learnings
2. Save new insights using the save_learning tool

## Workflow

1. Check Knowledge First
   - Before answering, search for relevant prior learnings
   - Apply any relevant insights to your response

2. Gather Information
   - Use YFinance tools for market data
   - Combine with your knowledge base insights

3. Save Valuable Insights
   - If you discover something reusable, save it with save_learning
   - The user will be asked to confirm before it's saved
   - Good learnings are specific, actionable, and generalizable

## What Makes a Good Learning

- Specific: "Tech P/E ratios typically range 20-35x" not "P/E varies"
- Actionable: Can be applied to future questions
- Reusable: Useful beyond this one conversation

Don't save: Raw data, one-off facts, or obvious information.\
"""


model = Ollama(
    id=DEFAULT_LLM_MODEL
)
agent_db = SqliteDb(db_file="tmp/agents.db")

learnings_kb = Knowledge(
    name="Agent Learnings",
    vector_db=ChromaDb(
        name="learnings",
        collection="learnings",
        path="tmp/chromadb",
        persistent_client=True,
        search_type=SearchType.hybrid,
        hybrid_rrf_k=60,
        embedder=OllamaEmbedder(id="llama3.2", dimensions=3072),
    ),
    max_results=5,
    contents_db=agent_db,
)


# ---------------------------------------------------------------------------
# Custom Tool: Save Learning
# ---------------------------------------------------------------------------
@tool(requires_confirmation=True)
def save_learning(title: str, learning: str) -> str:
    """
    Save a reusable insight to the knowledge base for future reference.

    Args:
        title: Short descriptive title (e.g., "Tech stock P/E benchmarks")
        learning: The insight to save — be specific and actionable

    Returns:
        Confirmation message
    """
    # Validate inputs
    if not title or not title.strip():
        return "Cannot save: title is required"
    if not learning or not learning.strip():
        return "Cannot save: learning content is required"

    # Build the payload
    payload = {
        "title": title.strip(),
        "learning": learning.strip(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save to knowledge base
    learnings_kb.insert(
        name=payload["title"],
        text_content=json.dumps(payload, ensure_ascii=False),
        reader=TextReader(),
        skip_if_exists=True,
    )

    return f"Saved: '{title}'"


finance_agent = Agent(
    name="Finance Agent",
    model=model,
    instructions=instructions,
    tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,),
            DuckDuckGoTools(fixed_max_results=2), 
            save_learning],
    db=agent_db,
    knowledge=learnings_kb,
    search_knowledge=True,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
)

if __name__ == "__main__":
    console = Console()
    console.print("Chat with mistral until entering an empty question or bye")
    done = False
    agent = finance_agent
    while not done:
        question = Prompt.ask("Question >", default="")
        if not question or "bye" in question:
            done = True
        else:
            run_response = agent.run(question)
            if run_response.content:
                pprint.pprint_run_response(run_response)
            console.print(run_response)
            if run_response.active_requirements:
                for requirement in run_response.active_requirements:
                    if requirement.needs_confirmation:
                        console.print(
                            f"\n[bold yellow]Confirmation Required[/bold yellow]\n"
                            f"Tool: [bold blue]{requirement.tool_execution.tool_name}[/bold blue]\n"
                            f"Args: {requirement.tool_execution.tool_args}"
                        )
                        choice = Prompt.ask(
                            "Do you want to continue?",
                            choices=["y", "n"],
                            default="y",
                        ).strip().lower()
                        if choice == "n":
                            requirement.reject()
                            console.print("[red]Rejected[/red]")
                        else:
                            requirement.confirm()
                            console.print("[green]Approved[/green]")
                run_response = agent.continue_run(
                    run_id=run_response.run_id,
                    requirements=run_response.requirements,
                )
                pprint.pprint_run_response(run_response)


