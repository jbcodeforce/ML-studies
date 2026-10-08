"""
Agno agent to get last news from different search engines for Flink
Updated 10/04/2026
"""

from agno.agent import Agent, RunOutput
from agno.models.openai.like import OpenAILike
from agno.tools.websearch import WebSearchTools
from agents.agno.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE, DEFAULT_LLM_API_KEY
from agno.utils.log import log_debug
from agno.utils.pprint import pprint_run_response
instructions = """
Search the web to find accurate and up-to-date information.
"""


model = OpenAILike(
    id=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_LLM_BASE_URL,
    temperature=DEFAULT_LLM_TEMPERATURE,
    api_key=DEFAULT_LLM_API_KEY,
)

_agent = Agent(
    name="Search information for flink",
    model=model,
    instructions=instructions,
    tools=[WebSearchTools(
        backend="google",
        fixed_max_results=3,
    )],
    markdown=True,
)



if __name__ == "__main__":
    # -- Tracing Setup --
    try:
        log_debug('Tracing enabled — Flink news search calls will be tracked')
    except ImportError:
        log_debug('Tracing disabled: OpenTelemetry not installed ' +\
                  '(pip install opentelemetry-api opentelemetry-sdk openinference-instrumentation-agno)')
    except Exception as exc:
        from agno.utils.log import log_debug
        log_debug(f'Tracing setup failed: {exc}')
    # ---------------------------------

    print('\n'+"="*60+'\n')
    print(f"Chat for web searchwith {DEFAULT_LLM_MODEL} until entering an empty question")
    print(f"Base URL: {DEFAULT_LLM_BASE_URL}")
    print('Example of question: What are the latest information of apache Flink vendors like Ververica, Cloudera, AWS or Confluent.')
    done = False
    while not done:
        print("Question >:")
        question = input()
        if not question or 'bye' in question:
            done = True
        else:
            response: RunOutput = _agent.run(question)
            pprint_run_response(response, markdown=True)
            print("\n\n")