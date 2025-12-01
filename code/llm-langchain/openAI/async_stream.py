from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import asyncio

load_dotenv("../../.env")

async def call_llm_astream():
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    parser = StrOutputParser()
    chain = prompt | model | parser
    # chain.astream is a generator yielding output
    async for chunk in chain.astream({"topic": "parrot"}):
        print(chunk, end="|", flush=True)


async def test_stream():
    await call_llm_astream()

asyncio.run(test_stream())