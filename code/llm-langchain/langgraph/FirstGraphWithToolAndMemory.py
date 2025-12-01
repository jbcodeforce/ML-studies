from typing import Literal, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState


from dotenv import load_dotenv

load_dotenv("../../.env")

print("-" * 30 )
print("----- Graph Demo -----")
print("-" * 30 )

class State(TypedDict):
    messages: Annotated[list, add_messages]

print("\n1- Define the tools and tool node")

tools = [TavilySearchResults(max_results=2)]
tool_node = ToolNode(tools)

print("\n2- Define the LLM model, selecting a model that support using tools")
llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    msg = state["messages"]  # [HumanMessage(content='What is a...)]
    rep = llm_with_tools.invoke(msg)  # AIMessage(content=[{'id': 'toolu_01KhuK3Hog', 'input': {'query': 'athena decision system'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}], 
    return {"messages": [rep]}

print("\n3- Build the Graph and function to assess node transition")

# Define a new graph
workflow = StateGraph(State)

# Define the two nodes we will cycle between
workflow.add_node("agent", chatbot)
workflow.add_node("tools", tool_node)

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    tools_condition,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')
workflow.add_edge(START,"agent")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


def send_msg(user_input: str, trd: str):
    config = {"configurable": {"thread_id": trd}}
    # The config is the **second positional argument** to stream() or invoke()!
    events = app.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()




send_msg("Hi there! My name is Will.", "1")
send_msg("Remember my name?","1")
send_msg("Remember my name?","2")

print(app.get_state( config = {"configurable": {"thread_id": "1"}}))