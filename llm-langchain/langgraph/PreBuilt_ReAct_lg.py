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


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
tools = [get_weather]
model = ChatOpenAI(temperature=0)
memory = MemorySaver()
graph = create_react_agent(model, tools=tools, checkpointer=memory)

inputs = {"messages": [("user", "what is the weather in sf")]}
print_stream(graph.stream(inputs, stream_mode="values"))