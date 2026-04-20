"""
Daily AI News Search Summary
============================

This workflow searches for the latest AI news and summarizes the key points.
"""
import asyncio
from textwrap import dedent
from typing import AsyncIterator
from agno.db.sqlite import SqliteDb

from agno.workflow import Workflow
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.run.workflow import WorkflowRunOutputEvent, WorkflowRunEvent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.websearch import WebSearchTools
from agno.agent import Agent
from agno.team import Team
from agno.models.ollama import Ollama


DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "gemma4:26b"
DEFAULT_LLM_TEMPERATURE = 0.4

_model = Ollama(
    id=DEFAULT_LLM_MODEL
)
# ---------------------------------------------------------------------------
# Create Agents
# ---------------------------------------------------------------------------
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=_model,
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
)

web_agent = Agent(
    name="Web Agent",
    model=_model,
    tools=[WebSearchTools()],
    role="Search the web for the latest news and trends",
)

content_planner = Agent(
    name="Content Planner",
    model=_model,
    instructions=[
        "Plan a content schedule over 4 weeks for the provided topic and research content",
        "Ensure that I have posts for 3 posts per week",
    ],
)

writer_agent = Agent(
    name="Writer Agent",
    model=_model,
    instructions="Write a blog post on the topic",
)

# ---------------------------------------------------------------------------
# Create Team
# ---------------------------------------------------------------------------
# Team needs an explicit model for coordination (routing). Without it, agno
# defaults to OpenAIChat and requires OPENAI_API_KEY.
research_team = Team(
    name="Research Team",
    model=_model,
    members=[hackernews_agent, web_agent],
    instructions="Research tech topics from Hackernews and the web",
)


# ---------------------------------------------------------------------------
# Define Steps
# ---------------------------------------------------------------------------
research_step = Step(
    name="Research Step",
    team=research_team,
)

content_planning_step = Step(
    name="Content Planning Step",
    agent=content_planner,
)



async def prepare_input_for_web_search(
    step_input: StepInput,
) -> AsyncIterator[StepOutput]:
    topic = step_input.input
    content = dedent(
        f"""\
        I'm writing a blog post on the topic
        <topic>
        {topic}
        </topic>

        Search the web for atleast 10 articles\
        """
    )
    yield StepOutput(content=content)


async def prepare_input_for_writer(step_input: StepInput) -> AsyncIterator[StepOutput]:
    topic = step_input.input
    research_team_output = step_input.previous_step_content
    content = dedent(
        f"""\
        I'm writing a blog post on the topic:
        <topic>
        {topic}
        </topic>

        Here is information from the web:
        <research_results>
        {research_team_output}
        </research_results>\
        """
    )
    yield StepOutput(content=content)


# ---------------------------------------------------------------------------
# Create Workflows
# ---------------------------------------------------------------------------


blog_post_workflow = Workflow(
    name="Blog Post Workflow",
    description="Automated blog post creation from Hackernews and the web",
    db=SqliteDb(
        session_table="workflow_session",
        db_file="../tmp/workflow.db",
    ),
    steps=[
        prepare_input_for_web_search,
        research_team,
        prepare_input_for_writer,
        writer_agent,
    ],
)




async def stream_run_events() -> None:
    events: AsyncIterator[WorkflowRunOutputEvent] = blog_post_workflow.arun(
        input="AI agent frameworks 2025",
        markdown=True,
        stream=True,
        stream_events=True,
    )
    async for event in events:
        if event.event == WorkflowRunEvent.condition_execution_started.value:
            print(event)
            print()
        elif event.event == WorkflowRunEvent.condition_execution_completed.value:
            print(event)
            print()
        elif event.event == WorkflowRunEvent.workflow_started.value:
            print(event)
            print()
        elif event.event == WorkflowRunEvent.step_started.value:
            print(event)
            print()
        elif event.event == WorkflowRunEvent.step_completed.value:
            print(event)
            print()
        elif event.event == WorkflowRunEvent.workflow_completed.value:
            print(event)
            print()


# ---------------------------------------------------------------------------
# Run Workflow
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Use Hacker News and the web to research the topic and write a blog post")
    # Async Streaming   
    query=input("Enter a query: ")
    asyncio.run(
        blog_post_workflow.aprint_response(
            input=query,
            markdown=True,
            stream=True,
        )
    )
    

    # Async Run Stream Events
    #asyncio.run(stream_run_events())