from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict, List, AsyncIterable
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv(dotenv_path="../../.env")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
tools = [search]
callback=AsyncIteratorCallbackHandler()  # return an async iterable
# Callback handlers are called throughout the lifecycle of a call to a chain, 
# starting with on_chain_start, ending with on_chain_end or on_chain_error. 
llm = ChatOpenAI(temperature=0.5, streaming=True, callbacks=[callback])
agent = create_tool_calling_agent(llm,tools,prompt)
ae=AgentExecutor(agent=agent,  tools=tools, verbose=True)

class ChatMessage(BaseModel):
    content: str


async def generate_llm_stream(content: str) -> AsyncIterable[str]:
    global ae
    # submit the coroutine to run in the background, switching between all other threads at await points
    # task is awaitable. the llm is now part of a thread and will get some time from the event loop
    task = asyncio.create_task(
        # asynchronous generation of the response
        ae.astream(input=content)
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



@app.post("/stream_chat")
async def stream_chat(msg: ChatMessage) -> Response:
    generator = generate_llm_stream(msg.content)
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server_wt_streaming:app", host="0.0.0.0", port=8000, reload=True)