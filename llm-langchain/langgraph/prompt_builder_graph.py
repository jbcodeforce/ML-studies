
from typing import Annotated
from typing_extensions  import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import MessageGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
import uuid
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List
from dotenv import load_dotenv


load_dotenv("../../.env")

"""
A chat bot that helps a user generate a prompt.
There are two separate states, gather requirements and Generate Prompt and the LLM decides when to transition between them.

Based on this tutorial: https://github.com/langchain-ai/langgraph/blob/main/examples/chatbots/information-gather-prompting.ipynb

"""
class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""
    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs

def define_gather_info_prompt():
    template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool"""
    return template

def get_messages_info(messages):
    return [SystemMessage(content=define_gather_info_prompt())] + messages

# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages):
    prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""

    tool_call = None
    other_msgs = []
    for m in messages:
        if _is_tool_call(m):
            tool_call = m.additional_kwargs["tool_calls"][0]["function"]["arguments"]
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs

def build_chains():
    """
    There are two nodes so two chains to support this processing.
    """
    llm = ChatOpenAI(temperature=0)
    llm_with_tool = llm.bind_tools([PromptInstructions])
    gather_info_chain = get_messages_info | llm_with_tool
    prompt_gen_chain = get_prompt_messages | llm
    return gather_info_chain, prompt_gen_chain

def get_state(messages):
    if _is_tool_call(messages[-1]):
        return "prompt"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    for m in messages:
        if _is_tool_call(m):
            return "prompt"
    return "info"

def define_graph():
    nodes = {k: k for k in ["info", "prompt", END]}
    workflow = MessageGraph()
    gather_info_chain, prompt_gen_chain = build_chains()
    workflow.add_node("info", gather_info_chain)
    workflow.add_node("prompt", prompt_gen_chain)
    workflow.add_conditional_edges("info", get_state, nodes)
    workflow.add_conditional_edges("prompt", get_state, nodes)
    workflow.set_entry_point("info")
    graph = workflow.compile()
    return graph


_graph= None

def get_or_build_graph():
    global _graph
    if not _graph:
        _graph=define_graph()
    return _graph

def send_user_msg(user_msg):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph=get_or_build_graph()
    return graph.stream([HumanMessage(content=user_msg)], config=config)


def text_chat():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while True:
        user = input("User (q/Q to quit): ")
        if user in {"q", "Q"}:
            print("AI: Byebye")
            break
        for output in graph.stream([HumanMessage(content=user)], config=config):
            print(output)
            if "__end__" in output:
                continue
            # stream() yields dictionaries with output keyed by node name
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value.content)
            print("\n---\n")


if __name__ == "__main__":
    text_chat()