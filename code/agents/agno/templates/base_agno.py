"""
Agent.... 
"""
import os
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from dotenv import load_dotenv
_env_file = os.getenv("ML_ENV_FILE")
load_dotenv(_env_file) if _env_file else load_dotenv()
# Default URL (e.g. mlx-llm-server or OpenAI-compatible proxy on 1337)
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:7999/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Ornith-1.0-9B-6bit")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "local_key")
AGENT_NAME="Agent"


INSTRUCTIONS = """
You are a ...
"""

_model =OpenAILike(
    id=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_LLM_BASE_URL,
    temperature=DEFAULT_LLM_TEMPERATURE,
    api_key=DEFAULT_LLM_API_KEY,  
)

_agent = Agent(
    model=_model,
    instructions=INSTRUCTIONS,
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
            run_response = _agent.run(question)
            print(run_response)

if __name__ == "__main__":
    print('\n'+"="*60+'\n')
    print(f"Agent {AGENT_NAME} with {DEFAULT_LLM_MODEL} until entering an empty question")
    print(f"Base URL: {DEFAULT_LLM_BASE_URL}")
    print('\n'+"="*60+'\n')
    repl()