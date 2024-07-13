from typing import AsyncGenerator, NoReturn, Annotated
from typing_extensions  import TypedDict
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint import MemorySaver

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import langchain


load_dotenv()

"""
Demonstrate streaming with human in the loop, and langgraph stream API.
The human in the loop is to get clarification via close question
This also demonstrate preparing data for a tool calling
"""

app = FastAPI()

langchain.debug=True

# ===================== Define tools for the application ==============
class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str

@tool
def search(query: str):
    """Call to surf the web."""
    return [
        f"I looked up: {query}. Result: It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    ]
    
@tool
def get_customer_risk_profile(lastname: str):
    """return the risk profile of the given customer knowing his last name. The risk factor is one of high, medium, low level"""
    return ["medium"]

tools = [get_customer_risk_profile, search]
tool_executor = ToolExecutor(tools)

sys_prompt =  ChatPromptTemplate.from_messages([
    ("system", """
     You are a investment risk assistant. Answer the user's questions 
     and use get_customer_risk_profile tool when you cannot get an answer on risk profile 
     or use search tool to get more recent data.
     """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user"),
])


model = sys_prompt | ChatOpenAI(temperature=0).bind_tools(tools + [AskHuman])



# =============== Define the graph to manage conversation with agents ==============
class State(TypedDict):
    """
    Keep the history of n messages
    """
    messages: Annotated[list, add_messages]

async def call_model(state):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": response}


def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    tool_call = last_message.tool_calls[0]
    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    response = tool_executor.invoke(action)
    tool_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    print(tool_message)
    return {"messages": [tool_message]}

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return "end"
    elif last_message.tool_calls[0]['name'] == "AskHuman":
        return "ask_human"
    else:
        print(last_message)
        return "continue"

# We define a fake node to ask the human
def ask_human(state):
    pass

def define_graph():
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    workflow.add_node("ask_human", ask_human)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            "ask_human": "ask_human",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")
    workflow.add_edge("ask_human", "agent")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])


graph=define_graph()
async def get_ai_response(message: str) -> AsyncGenerator[str, None]:
    all_content = ""
    inputs = [HumanMessage(content=message)]
    thread = {"configurable": {"thread_id": "2"}}
    print(inputs)
    async for event in graph.astream({"messages": inputs}, thread, stream_mode="values"):
        message =  event["messages"][-1]
        if message:
            all_content += message.content
            yield all_content

# ======================================== API And UI ============================
with open("index.html") as f:
    html = f.read()
    
@app.get("/")
async def web_app() -> HTMLResponse:
    """
    Web App
    """
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> NoReturn:
    """
    Websocket for AI responses
    """
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        # stream the response from the graph as soon as tokens are yielded
        async for text in get_ai_response(message):
            await websocket.send_text(text)


if __name__ == "__main__":
    print("=============== Langgraph streaming ===============")
    uvicorn.run(
        "main_lg:app",
        host="0.0.0.0",
        port=8010,
        #log_level="debug",
        reload=True,
    )