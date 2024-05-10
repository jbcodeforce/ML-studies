from typing import Annotated
from typing_extensions  import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import langchain
import os

class State(TypedDict):
    messages: Annotated[list, add_messages]


def define_model():
    return ChatOpenAI(temperature=0)

@tool
def search(query: str):
    """
    Call to search the web when you need weather information.
    """
    return [" it is sunny in Santa Clara"]


load_dotenv("../../.env")
langchain.debug=True
tools = [search]
tool_executor = ToolExecutor(tools)
model = define_model()
model.bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"
    
# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a ToolMessage
    tool_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}

# Define a new graph
workflow = StateGraph(State)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

memory = SqliteSaver.from_conn_string(":memory:")
# compiles it into a LangChain Runnable
# app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
app = workflow.compile(interrupt_before=["action"])
# Mock the conversation
thread = {"configurable": {"thread_id": "2"}}
inputs = [HumanMessage(content="hi! I'm bob")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

inputs = [HumanMessage(content="What did I tell you my name was?")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

inputs = [HumanMessage(content="what's the weather in santa clara (CA) now? use your tools if you do not know")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# Finish to trigger the end condition
for event in app.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()