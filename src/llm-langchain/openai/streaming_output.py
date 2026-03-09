from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path="../../.env")

llm = ChatOpenAI(temperature=0,stream=True)

def most_simple():
    global llm
    chunks = []
    for chunk in llm.stream("hello. when it will make sense to use langchain streaming"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)
    print("\n--------------")



most_simple()