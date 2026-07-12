
from typing import Annotated, List, Literal
import uuid
from dotenv import load_dotenv
from typing_extensions  import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.graph import MessageGraph, END, START

load_dotenv("../../.env")

"""
A chat bot that helps a user to generate a prompt.
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
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
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

def get_state(messages) -> Literal["add_tool_message", "info", "__end__"]:
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"


    
def define_graph():
    memory = SqliteSaver.from_conn_string(":memory:")
    workflow = MessageGraph()
    gather_info_chain, prompt_gen_chain = build_chains()
    workflow.add_node("info", gather_info_chain)
    workflow.add_node("prompt", prompt_gen_chain)
    @workflow.add_node
    def add_tool_message(state: list):
        return ToolMessage(
            content="Prompt generated!", tool_call_id=state[-1].tool_calls[0]["id"]
        )
    
    workflow.add_edge("add_tool_message", "prompt")
    workflow.add_conditional_edges("info", get_state)
    workflow.add_edge("prompt", END)
    workflow.add_edge(START, "info")
    graph = workflow.compile(checkpointer=memory)
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
    global _graph 
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        user = input("User (q/Q to quit): ")
        if user in {"q", "Q"}:
            print("AI: Byebye")
            break
        output = None
        for output in _graph.stream([HumanMessage(content=user)], config=config, stream_mode="updates"):
            last_message = next(iter(output.values()))
            last_message.pretty_print()
        if output and "prompt" in output:
                print("Done!")

if __name__ == "__main__":
    get_or_build_graph()
    text_chat()