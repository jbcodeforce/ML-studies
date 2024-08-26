
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import argparse

print(" The LLM is an earth environment expert so ask question about environment.")
parser = argparse.ArgumentParser(description='Simple text improvement with Llama3 ')
parser.add_argument('text', type=str, help='The text to improve')
args = parser.parse_args()


llm = ChatOllama(model="llama3.1", base_url='http://localhost:11434',)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a earth environment expert."),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print("--- stream from LLama3 via a simple LangChain chain")

for chunks in chain.stream({"input": args.text}):
    print(chunks)
