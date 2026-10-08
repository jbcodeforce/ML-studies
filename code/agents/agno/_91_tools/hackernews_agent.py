"""
Agno agent to get last news from Hacker News
Updated 09/30/2026
"""

from agno.agent import Agent, RunOutput
from agno.models.openai.like import OpenAILike
from agno.tools.hackernews import HackerNewsTools
from agents.agno.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE, DEFAULT_LLM_API_KEY
from agno.utils.log import log_debug
from agno.utils.pprint import pprint_run_response
instructions = """
Write a report on the topic. Output only the report.
"""


model = OpenAILike(
    id=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_LLM_BASE_URL,
    temperature=DEFAULT_LLM_TEMPERATURE,
    api_key=DEFAULT_LLM_API_KEY,
)

hackernews_agent = Agent(
    name="Hacker News Agent",
    model=model,
    instructions=instructions,
    tools=[HackerNewsTools(all=True)],
    add_datetime_to_context=True,
    markdown=True,
)



if __name__ == "__main__":
    # -- Tracing Setup --
    try:
        # Install: pip install opentelemetry-api opentelemetry-sdk openinference-instrumentation-agno

        log_debug('Tracing enabled — HackerNewsTools calls will be tracked')
    except ImportError:
        log_debug('Tracing disabled: OpenTelemetry not installed ' +\
                  '(pip install opentelemetry-api opentelemetry-sdk openinference-instrumentation-agno)')
    except Exception as exc:
        from agno.utils.log import log_debug
        log_debug(f'Tracing setup failed: {exc}')
    # ---------------------------------

    print('\n'+"="*60+'\n')
    print(f"Chat for Hacker News with {DEFAULT_LLM_MODEL} until entering an empty question")
    print(f"Base URL: {DEFAULT_LLM_BASE_URL}")
    print('Example of question: Trending startups and products.')
    done = False
    agent = hackernews_agent
    while not done:
        print("Question >:")
        question = input()
        if not question or 'bye' in question:
            done = True
        else:
            response: RunOutput = agent.run(question)
            pprint_run_response(response, markdown=True)
            print("\n\n")