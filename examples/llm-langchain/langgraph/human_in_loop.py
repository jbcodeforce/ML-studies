from typing import Annotated
from typing_extensions  import TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode,  tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, ToolMessage
load_dotenv("../../.env")


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def search(query_about_weather: str):
    """
    Search the web when you need weather information.
    """
    return [" it is sunny in Santa Clara"]


def define_model_with_tools(tools):
    sys_prompt =  ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions and use tools when you cannot get an answer on recent data."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user"),
        ])

    #llm = ChatOpenAI(temperature=0, model="gpt-4o")
    llm = ChatAnthropic(model="claude-3-haiku-20240307")
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools 

  

def define_graph():
    tavily = TavilySearchResults(max_results=2)
    tools = [tavily]
    model=define_model_with_tools(tools)

    def call_model(state: State):
        messages = state["messages"]
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


    tool_node = ToolNode(tools=tools)
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", 'agent')
    workflow.set_entry_point("agent")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])
    return app

if __name__ == "__main__":
    app=define_graph()
    config = {"configurable": {"thread_id": "2"}}
    user_input = "I'm learning LangGraph. Could you do some research on it for me?"
    for event in app.stream({"messages": [("user", user_input)]}, config):
        if "messages" in event:
            event["messages"][-1].pretty_print()

    snapshot = app.get_state(config)
    print(f"Next step is: {snapshot.next}")
    # interrupted at the tools level
    existing_message = snapshot.values["messages"][-1]
    print(f" tools calls is {existing_message.tool_calls}")
    existing_message.pretty_print()
    # The simplest thing the human can do is just let the graph continue executing.
    #     events = app.stream(None, config, stream_mode="values")
    # OR give the answer
    answer = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs."
    )
    new_messages = [
        # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
        ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
        # And then directly "put words in the LLM's mouth" by populating its response.
        AIMessage(content=answer),
    ]
    new_messages[-1].pretty_print()
    app.update_state(
        # Which state to update
        config,
        # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
        # to the existing state. We will review how to update existing messages in the next section!
        {"messages": new_messages},
    )
    print("\n\nLast 2 messages;")
    print(app.get_state(config).values["messages"][-2:])
    