"""
Demonstrate streaming response from the agent with chatbot using text
"""

from dotenv import load_dotenv
load_dotenv("../../.env")
import os
import asyncio
from typing import Literal
#from langchain import hub
from langchain_core.messages import  HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode



from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults



tool = TavilySearchResults(api_key=os.environ.get("TAVILY_API_KEY"), max_results=4)
tools = [tool]
tool_node = ToolNode(tools)

model = ChatOpenAI(temperature=0).bind_tools(tools)



class State(TypedDict):
    """
    Keep the history of n messages
    """
    messages: Annotated[list, add_messages]

async def call_model(state):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": response}

def should_continue(state: State) -> Literal["__end__", "tools"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "tools"


def define_graph():
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()
      
  
    
async def text_chat(graph):
    config = {"configurable": {"thread_id": 1}}
    
    while True:
        user_msg = input("User (q/Q to quit): ")
        if user_msg in {"q", "Q"}:
            print("AI: Byebye")
            break
        async for event in graph.astream_events({"messages": [HumanMessage(content=user_msg)]}, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="|",flush=True)
            elif kind == "on_tool_start":
                print("--")
                print(
                    f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
                )
            elif kind == "on_tool_end":
                print(f"Done tool: {event['name']}")
                print(f"Tool output was: {event['data'].get('output')}")
                print("--")


if __name__ == "__main__":
    graph=define_graph()
    asyncio.run(text_chat(graph))