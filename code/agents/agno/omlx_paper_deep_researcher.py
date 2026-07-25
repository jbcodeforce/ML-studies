"""
Based on Agno seek demo, create a deep researcher agent.
Read a research paper to extract main information and insights
then build a learning path to learn more.
"""
import os
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.websearch import WebSearchTools
from agno.tools.file import FileTools
from rich.console import Console
from rich.prompt import Prompt
from agno.media import File
from dotenv import load_dotenv

_env_file = os.getenv("ML_ENV_FILE")
load_dotenv(_env_file) if _env_file else load_dotenv()

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:7999/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-27B-4bit")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "local_key")


# knowledge managfement
model =OpenAILike(
    id=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_LLM_BASE_URL,
    temperature=DEFAULT_LLM_TEMPERATURE,
    api_key=DEFAULT_LLM_API_KEY,  
)

paper_analysis_agent = Agent(
    model=model,
    tools=[WebSearchTools(),FileTools(enable_list_files=True, enable_read_file=True, enable_search_files=True)],
    instructions="""
    Summarize the paper specified as file name or URL and provide a learning path to learn more. 
    Use web search when URL.
    """,
    markdown=True
)


if __name__ == "__main__":
    print('\n'+"="*60+'\n')
    print(f"Chat for market finanical analysis with {DEFAULT_LLM_MODEL} until entering an empty question")
    print(f"Base URL: {DEFAULT_LLM_BASE_URL}")
    print('\n'+"="*60+'\n')
    console = Console()
    console.print("Deep research agents, first specify a url or file to read")
    print("You will see the response is part of long trace with messages.")
    console.print(f"Example of query: summarize me this paper: https://arxiv.org/html/1706.03762v7 and references related code sources")
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