"""
Learning Machines with oMLX chat + Ollama extraction
====================================================
Agno LearningMachine saves user profile and memories via tool calls during
background extraction (ALWAYS mode) or via agent tools (AGENTIC mode).

oMLX/Codestral often answers chat fine but does not return tool_calls, so
learning extraction saves nothing and session 2 gets no <user_memory> context.
Use a tool-capable Ollama model on LearningMachine.model for extraction only.

See README.md — MLX/oMLX tool calling limitations.
"""

import os
import sys

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.learn import (
    LearningMachine,
    LearningMode,
    UserMemoryConfig,
    UserProfileConfig,
)
from agno.models.ollama import Ollama
from agno.models.openai.like import OpenAILike

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:7999/v1"
# oMLX model id matches the directory name under OMLX_MODEL_DIR (see GET /v1/models)
DEFAULT_LLM_MODEL = "Codestral-22B-v0.1-4bit"
DEFAULT_LLM_TEMPERATURE = 0.4
DEFAULT_LLM_API_KEY = "localkey"

# Tool-capable model for learning extraction (profile + memory stores)
# Must return tool_calls during extraction (mistral:7b-instruct often narrates tools in text instead).
DEFAULT_LEARNING_OLLAMA_MODEL = os.getenv("LEARNING_OLLAMA_MODEL", "qwen3.6:35b-a3b")

INSTRUCTIONS = """\
You may receive <user_profile> and <user_memory> blocks in your system context.
Those are facts about this user from prior sessions — use them naturally.
When asked what you know about the user, answer from that context.
Do not claim you have no memory if profile or memory blocks are present.
"""

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
db = SqliteDb(db_file="tmp/agents.db")

chat_model = OpenAILike(
    id=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_LLM_BASE_URL,
    temperature=DEFAULT_LLM_TEMPERATURE,
    api_key=DEFAULT_LLM_API_KEY,
)

# Extraction runs a second LLM call with tools; must use a model that returns tool_calls.
learning_model = Ollama(id=DEFAULT_LEARNING_OLLAMA_MODEL)


def _learning_machine(mode: LearningMode = LearningMode.ALWAYS) -> LearningMachine:
    if mode == LearningMode.ALWAYS:
        return LearningMachine(
            db=db,
            model=learning_model,
            user_profile=True,
            user_memory=True,
        )
    return LearningMachine(
        db=db,
        model=learning_model,
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
    )


def agentic_learner():
    return Agent(
        model=chat_model,
        db=db,
        learning=_learning_machine(LearningMode.AGENTIC),
        instructions=INSTRUCTIONS,
        markdown=True,
    )


def basic_agent():
    return Agent(
        model=chat_model,
        db=db,
        learning=_learning_machine(LearningMode.ALWAYS),
        instructions=INSTRUCTIONS,
        markdown=True,
    )


def verify_learning_saved(lm: LearningMachine, user_id: str) -> bool:
    """Return True if profile or memories exist for user_id after session 1."""
    profile = lm.user_profile_store.recall(user_id=user_id) if lm.user_profile_store else None
    memories = lm.user_memory_store.recall(user_id=user_id) if lm.user_memory_store else None
    has_profile = bool(profile and (getattr(profile, "name", None) or getattr(profile, "preferred_name", None)))
    memory_list = getattr(memories, "memories", None) if memories else None
    has_memories = bool(memory_list and len(memory_list) > 0)
    return has_profile or has_memories
# ---------------------------------------------------------------------------
# Run Demo
# 1- basic
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    user_id = "jerome@example.com"

    agent = basic_agent()
    # Session 1: Agent decides what to save via tool calls
    print("\n--- Session 1: Agent uses tools to save profile and memories ---\n")
    agent.print_response(
        "Hi! I'm Jerome. I work at Confluent.io as a data streaming architect. "
        "I prefer concise responses without too much explanation.",
        user_id=user_id,
        session_id="session_1",
        stream=True,
    )
    lm = agent.learning_machine
    lm.user_profile_store.print(user_id=user_id)
    lm.user_memory_store.print(user_id=user_id)

    if not verify_learning_saved(lm, user_id):
        print(
            "\n[warning] No profile or memories were saved after session 1.\n"
            "Learning extraction requires tool_calls. Ensure Ollama is running "
            f"(`ollama serve`) and model '{DEFAULT_LEARNING_OLLAMA_MODEL}' is pulled.\n"
            "Session 2 will not remember the user until extraction succeeds.\n",
            file=sys.stderr,
        )

    # Session 2: New session - agent remembers
    print("\n--- Session 2: Agent remembers across sessions ---\n")
    agent.print_response(
        "What do you know about me?",
        user_id=user_id,
        session_id="session_2",
        stream=True,
    )