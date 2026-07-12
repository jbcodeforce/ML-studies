from dotenv import load_dotenv
from typing import Literal
load_dotenv("../.env")

from langchain_core.tools import tool
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults


from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

MODEL_NAME="mistral-small-latest"

# ================ most likely in a separate module

@tool
def get_user_info(user_id: str):
    """Call when you need to get information of a user given it user_id"""
    return "The user is Bob the Builder"

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
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

web_search_tool = TavilySearchResults(k=3)

tools = [web_search_tool, get_user_info, get_weather]

mistral_model = ChatMistralAI(model=MODEL_NAME)
#mistral_model.bind_tools(tools)
#tool_executor = ToolExecutor(tools)

    

memory = SqliteSaver.from_conn_string(":memory:")
graph = create_react_agent(mistral_model, tools=tools, checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [("user", "What's the weather in NYC?")]}
print_stream(graph.stream(inputs, config=config, stream_mode="values"))

inputs = {"messages": [("user", "Who is the user with user_id = F05?")]}
print_stream(graph.stream(inputs, config=config, stream_mode="values"))