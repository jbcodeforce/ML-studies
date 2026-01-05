from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


"""
The simplest chain with OpenAI. Get the API key from .env file
"""
print(" --- Load environment variables, like the OPENAI_API_KEY")
load_dotenv(dotenv_path="../../.env")

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print("--- call openAI via a simple LangChain chain")
print("\tquery: how can LangSmith help with testing?")

resp=chain.invoke({"input": "how can LangSmith help with testing?"})
# The results should includes some hallucinations
print(resp)
print("\n\n---- should give wrong results.")