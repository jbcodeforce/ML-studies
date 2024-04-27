import json
from pydantic import BaseModel
import requests

class ChatMessage(BaseModel):
    content: str


url = "http://localhost:6667"



headers = {"Content-type": "application/json"}
def call_tool_access_directly(path: str, item: str):
    global url
    resp=requests.get(f"{url}{path}{item}")
    print(resp.json())

def call_sync(path: str, data: ChatMessage):
    global url
    resp=requests.post(f"{url}{path}", data=data.model_dump_json(), headers=headers)
    print(resp.json())

def call_async(path: str, data: ChatMessage ):
    global url
    with requests.post(f"{url}{path}", data=data.model_dump_json(), headers=headers, stream=True) as r:
        for chunk in r.iter_content(1024):
            print(chunk)


data = ChatMessage(content="what is on the bed?")
# Test access to the tool directly without going to LLM -> works
# call_tool_access_directly("/items_via_tool/","bed")

# Test with sync tool and LLM -> works
# Example of output
# {'input': 'what is on the bed?', 
#'output': 'On the bed, there are socks, shoes, and dust bunnies.'}
# call_sync("/chat/", data)

# Test async streaming with backend being a chain
call_async("/chat_chain/", data)

# Test async streaming with backend being a agent executor
#call_async("/stream_chat/", data)