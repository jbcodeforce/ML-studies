from typing import Annotated, Literal
from dotenv import load_dotenv
import json
load_dotenv("../../.env")
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic

from langchain_community.tools.tavily_search import TavilySearchResults

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        """
        {'messages': [HumanMessage(content='What is athena decision system?', id='e11d2bb5bf'), 
                      AIMessage(content=[{'id': 'toolXKcez', 
                                          'input': {'query': 'athena decision system'}, 
                                          'name': 'tavily_search_results_json', 
                                          'type': 'tool_use'}
                                          ], 
                    response_metadata=...]}
        """
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            """
            [{'url': 'https://athenadecisionsystems.github.io/athena-owl-core/', 
             'content': 'Athena Decision Systems is here t.......'}, 
             {'url': 'https://athenadecisions.com/', 'content': 'At Athena Decision Systems, we want to ...'}
             ]
            """
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    

def chatbot(state: State):
    msg = state["messages"]  # [HumanMessage(content='What is athena decision system?', id='802864c...15aa1')]
    rep = llm_with_tools.invoke(msg)  # AIMessage(content=[{'id': 'toolu_01KhuK3Hog', 'input': {'query': 'athena decision system'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}], response_metadata={'id': 'msg_01DX7uiRdkKJYdpCyNT2GmR8', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 374, 'output_tokens': 61}}, id='run-5f41e11b-a68b-4fe8-9256-7b8ba93be5ca-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'athena decision system'}, 'id': 'toolu_01KhujdUSZjwvFh1AnwK3Hog', 'type': 'tool_call'}], usage_metadata={'input_tokens': 374, 'output_tokens': 61, 'total_tokens': 435})
    return {"messages": [rep]}

def route_tools(
    state: State,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"



tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", "__end__": "__end__"},
)

graph = graph_builder.compile()

def chat_with_human():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        for event in graph.stream({"messages": ("user", user_input)}):
            for value in event.values():
                if isinstance(value["messages"][-1], BaseMessage):
                    print("Assistant:", value["messages"][-1].content)

if __name__ == "__main__":
    rep=graph.invoke({"messages": ("user", "What is athena decision system?")})
    print(rep)