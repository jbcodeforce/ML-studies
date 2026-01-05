from dotenv import load_dotenv

load_dotenv("../../.env")
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from langgraph.checkpoint import MemorySaver

print("----- First Graph Demo -----")
print("-" * 40 )
print("\n1- Define the tools and tool node")

@tool
def get_weather(city: Literal["sf", "nyc"]):
    """Use this tool to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


def print_stream(graph, inputs, config):
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
tools = [get_weather]
model = ChatOpenAI(temperature=0)
memory = MemorySaver()
graph = create_react_agent(model, tools=tools, interrupt_before=["tools"],  checkpointer=memory)
thread = {"configurable": {"thread_id": "2"}}
inputs = {"messages": [("user", "what is the weather in sf")]}
print_stream(graph, inputs, thread)
snapshot = graph.get_state(thread)
print(">>> Flow interrupted, next step is: ", snapshot.next)
print_stream(graph, None, thread)
inputs2 = {"messages": [("user", "Cool, so then should I go biking today?")]}
print_stream(graph, inputs2, thread)


# user input is None

