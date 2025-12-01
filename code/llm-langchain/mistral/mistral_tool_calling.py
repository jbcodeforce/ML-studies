"""
Tool calling with mistral new models
"""
import pandas as pd
import json, functools, os
from pydantic import BaseModel
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient

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


def define_graph():
    return None
    

if __name__ == "__main__":
    print("=" * 20 + " mockup a dialog with transaction agent " + "=" *20 )
    messages = [
            ChatMessage(role="user", content="What's the status of my transaction T1001?")
        ]
    model = "mistral-large-latest"
    api_key = os.getenv("MISTRAL_API_KEY")
    tools=define_tool_specifications()
    client = MistralClient(api_key=api_key)
    response = client.chat(model=model, messages=messages, tools=tools, tool_choice="auto")
    print(response)
    """
    id='e097a3ca51454e2baf65efa25d4b4eb3' 
    object='chat.completion' 
    created=1720576206 
    model='mistral-large-latest' 
    choices=[ChatCompletionResponseChoice(index=0, 
        message=ChatMessage(role='assistant', 
                            content='', 
                            name=None, 
                            tool_calls=[ToolCall(id='jB9F7wxP1', 
                                                 type=<ToolType.function: 'function'>, 
                                                 function=FunctionCall(name='retrieve_payment_status', 
                                                                       arguments='{"transaction_id": "T1001"}')
                                                )], 
                            tool_call_id=None), 
        finish_reason=<FinishReason.tool_calls: 'tool_calls'>)
        ] 
    usage=UsageInfo(prompt_tokens=166, total_tokens=196, completion_tokens=30)
    """
    messages.append(response.choices[0].message)
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_params = json.loads(tool_call.function.arguments)
    print("\nfunction_name: ", function_name, "\nfunction_params: ", function_params)
    function_result = names_to_functions[function_name](**function_params)
    # generate final answer
    messages.append(ChatMessage(role="tool", name=function_name, content=function_result, tool_call_id=tool_call.id))
    response = client.chat(model=model, messages=messages)
    print(response.choices[0].message.content)