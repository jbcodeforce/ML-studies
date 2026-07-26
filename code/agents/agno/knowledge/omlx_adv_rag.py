"""
This is the advanced RAG built from agno cookbook to jumpstart a simple knowledge manager agent.

- use system prompt for things filtered out of the vector database
- use agentic RAG, so agent is deciding to search or not.
"""
import os
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from dotenv import load_dotenv
from prepare_vs import prepare_knowledge_base

_env_file = os.getenv("ML_ENV_FILE")
load_dotenv(_env_file) if _env_file else load_dotenv()
# Default URL (e.g. mlx-llm-server or OpenAI-compatible proxy on 1337)
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:7999/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Ornith-1.0-9B-6bit")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "local_key")
AGENT_NAME="Agent"


INSTRUCTIONS = [
        "Always search your knowledge base before answering.",
        "Include sources in your response.",
    ]

_model =OpenAILike(
    id=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_LLM_BASE_URL,
    temperature=DEFAULT_LLM_TEMPERATURE,
    api_key=DEFAULT_LLM_API_KEY,  
)

kb = prepare_knowledge_base()

_agent = Agent(
    model=_model,
    knowledge=kb,
    instructions=INSTRUCTIONS,
    search_knowledge=True,
    enable_agentic_knowledge_filters=True,
    markdown=True
)


def repl():
    """
    Read-Eval-Print Loop
    """
    done = False
    while not done:
        print("Question >:")
        question = input()
        if not question or "bye" in question:
            done = True
        else:
            _agent.print_response(question, stream=True)


if __name__ == "__main__":
    print('\n'+"="*60+'\n')
    print(f"Agent {AGENT_NAME} with {DEFAULT_LLM_MODEL} until entering an empty question")
    print(f"Base URL: {DEFAULT_LLM_BASE_URL}")
    print('\n'+"="*60+'\n')
    repl()