"""
Tool calling with mistral  and langgraph for orchestration
"""
import pandas as pd
import json, functools, os, operator
from typing import Annotated, TypedDict
from pydantic import BaseModel
from mistralai.models.chat_completion import ChatMessage
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from mistralai.client import MistralClient

from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv("../../.env")

# mockup data
data =  {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}
df = pd.DataFrame(data)

# ================= Define tools ====================
def retrieve_payment_status(df, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values: 
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'transaction id not found.'})

def retrieve_payment_date(df, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values: 
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
    return json.dumps({'error': 'transaction id not found.'}) 

def define_tool_specifications():
    tools = [ {
                "type": "function",
                "function": {
                    "name": "retrieve_payment_status",
                    "description": "Get payment status of a transaction",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "transaction_id": {
                                "type": "string",
                                "description": "The transaction id.",
                            }
                        },
                        "required": ["transaction_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_payment_date",
                    "description": "Get payment date of a transaction",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "transaction_id": {
                                "type": "string",
                                "description": "The transaction id.",
                            }
                        },
                        "required": ["transaction_id"],
                    },
                },
            }
            ]
    return tools

names_to_functions = {
    'retrieve_payment_status': functools.partial(retrieve_payment_status, df=df),
    'retrieve_payment_date': functools.partial(retrieve_payment_date, df=df)
}

class AgentState(TypedDict):
    """
    Accumulate the messages of the conversation
    """
    messages: Annotated[list[ChatMessage], operator.add]

def call_llm(state: AgentState):
    model = "mistral-large-latest"
    api_key = os.getenv("MISTRAL_API_KEY")
    client = MistralClient(api_key=api_key)
    tools=define_tool_specifications()
    message = state['messages'][-1]
    print(message)
    other_fmt_message=ChatMessage(role="user", content=message.content)
    response = client.chat(model=model, messages=[other_fmt_message], tools=tools, tool_choice="auto")
    return { "messages": [response.choices[0].message]}

def take_action(state: AgentState):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        function_name = t.function.name
        function_params = json.loads(t.function.arguments)
        print("\nfunction_name: ", function_name, "\nfunction_params: ", function_params)
        if not function_name in names_to_functions:
            print("\n ....bad tool name....")
            result = "bad tool name, retry"
        else:
            function_result = names_to_functions[function_name](**function_params)
        results.append(ChatMessage(role="tool", name=function_name, content=function_result, tool_call_id=t.id))
    return {"messages": results}

def exists_action(state: AgentState):
    result = state['messages'][-1]
    if not result.tool_calls:
        return False
    return len(result.tool_calls) > 0
    
def define_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("action", take_action)
    graph.add_conditional_edges(
        "llm",
        exists_action,
        {True: "action", False: END}
    )
    graph.add_edge("action", "llm")
    graph.set_entry_point("llm")
    graph = graph.compile()
    return graph
    

if __name__ == "__main__":
    print("=" * 20 + " mockup a dialog with transaction query with langgraph " + "=" *20 )
    messages = [
        HumanMessage(content="What's the status of my transaction T1001?")
        ]
    graph = define_graph()
    messages = graph.invoke({"messages": messages})
    print("=" * 20 + " final response " + "=" *20 )
    final_resp = messages["messages"][-1].content
    print(f"\n\n {final_resp}")