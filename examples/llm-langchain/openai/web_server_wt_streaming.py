from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel
from typing import Any, Dict, List, AsyncIterable
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.tools import tool
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults

"""
The goal is to demonstrate how to stream with agent executor using a FastAPI endpoint. 

The /chat is synchronous so client waits to get the answer. 

the implementation uses tool calling. 
See also the client_stream.py code to test this server.
"""

load_dotenv(dotenv_path="../../.env")

def search_items(place: str) -> str:
    """Use this tool to look up which items are in the given place."""
    if "bed" in place:  # For under the bed
        return "socks, shoes and dust bunnies"
    if "shelf" in place:  # For 'shelf'
        return "books, pencils and pictures"
    else:  # if the agent decides to ask about a different place
        return "cat snacks"
    
# --- only one of the following tool will be used, depending if the model is invoked sync ir async
@tool
async def get_items(place: str) -> str:
    """Use this tool to look up which items are in the given place."""
    return search_items(place)

@tool
def get_items_sync(place: str) -> str:
    """Use this tool to look up which items are in the given place."""
    return search_items(place)



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use the tavily_search_results_json tool for recent information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

def define_async_agent():
    search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
    tools = [search, get_items]
    #callback=AsyncIteratorCallbackHandler()  # return an async iterable
    callback=StreamingStdOutCallbackHandler()
    # Callback handlers are called throughout the lifecycle of a call to a chain, 
    # starting with on_chain_start, ending with on_chain_end or on_chain_error. 
    llm = ChatOpenAI(temperature=0.5, streaming=True, callbacks=[callback])
    agent = create_tool_calling_agent(llm,tools,prompt)
    agent_executor=AgentExecutor(agent=agent,  tools=tools, verbose=True).with_config(
        {"run_name": "Agent"}
    )
    return agent_executor, callback

def define_sync_agent():
    search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
    tools = [search, get_items_sync]
    llm = ChatOpenAI(temperature=0.5)
    agent = create_tool_calling_agent(llm,tools,prompt)
    agent_executor=AgentExecutor(agent=agent,  tools=tools, verbose=True).with_config(
        {"run_name": "Agent"}
    )
    return agent_executor


async def send_message_using_chain(msg: str) -> AsyncIterable[str]:
    """
    
    """
    callback = AsyncIteratorCallbackHandler()
    llm = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )
    tools = [get_items_sync]
    parser = StrOutputParser()
    chain = prompt | llm | parser
    task = asyncio.create_task(
        chain.agenerate(messages={"input": msg})
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task

class ChatMessage(BaseModel):
    content: str

async def run_call(ae, query: str):
    # now query
    await ae.astream(inputs={"input": query})

async def generate_llm_stream(content: str):
    ae, callback = define_async_agent()
    # submit the coroutine to run in the background, switching between all other threads at await points
    # task is awaitable. the llm is now part of a thread and will get some time from the event loop
    task = asyncio.create_task(
        # asynchronous generation of the response
       run_call(ae,content)
    )

    # ensure there is no more token to process
    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task


# -------------------------------------------- Define API server and APIs
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/items_via_tool/{place}")
async def get_items_api(place: str) -> Response:
    """
    tool methods are async so need to invoke them asynchronously
    """
    return await get_items.ainvoke(place)


@app.get("/stream_items/{place}")
async def get_items_as_stream(place: str) -> Response:
    """
    Demonstrate tracing intermediate steps, and streaming back to client
    """
    chunks = []
    agent_executor = define_async_agent()
    async for chunk in agent_executor.astream(
        {"input": f"what's items are located on the {place}?"}
    ):
        chunks.append(chunk)
        print("------")
        print(chunk["messages"])


@app.post("/chat_chain")
async def chat_using_chain(message: ChatMessage):
    """
    Generate a stream of token, using a langchain chain: llm, prompt and tools
    """
    generator = send_message_using_chain(message.content)
    return StreamingResponse(generator, media_type="text/event-stream")

@app.post("/chat")
def chat_with_llm(msg: ChatMessage) -> Response:
    ae = define_sync_agent()
    return ae.invoke({"input" : msg.content})

@app.post("/stream_chat")
async def stream_chat(msg: ChatMessage) -> Response:
    """
    The streaming api. it uses a generator to get the 
    """
    # need to send our tokens to a generator function that feeds those tokens to FastAPI via a StreamingResponse object
    
    generator = generate_llm_stream(msg.content)
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("web_server_wt_streaming:app", host="0.0.0.0", port=6667, reload=True)