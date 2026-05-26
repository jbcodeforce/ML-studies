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
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:7999/v1"
# oMLX model id matches the directory name under OMLX_MODEL_DIR (see GET /v1/models)
DEFAULT_LLM_MODEL = "Codestral-22B-v0.1-4bit"
DEFAULT_LLM_TEMPERATURE = 0.4
DEFAULT_LLM_API_KEY = "localkey"

# knowledge managfement
model =OpenAILike(
    id=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_LLM_BASE_URL,
    temperature=DEFAULT_LLM_TEMPERATURE,
    api_key=DEFAULT_LLM_API_KEY,  
)

paper_analysis_agent = Agent(
    model=model,
    tools=[WebSearchTools(),{"type": "file_search"}],
    instructions="Summarize the paper and provide a learning path to learn more.",
    markdown=True,
    add_history_to_context=True,
)


if __name__ == "__main__":
    console = Console()
    console.print("Deep research agents, first specify a url or file to read")
    console.print(f"Example of query: ")
    done = False
    while not done:
        question = Prompt.ask("Question >", default="")
        if not question or "bye" in question:
            done = True
        elif question.startswith("file:"):

            file_path = question.split("file:")[1]
            file = File(filepath=file_path)
            run_response = paper_analysis_agent.run(file)
            console.print(run_response)
        else:
            run_response = paper_analysis_agent.run(question)
            console.print(run_response) 