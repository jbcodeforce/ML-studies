
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import argparse

print(" The LLM is english expert to fix your english text.")
parser = argparse.ArgumentParser(description='Simple text improvement with Llama3 ')
parser.add_argument('text', type=str, help='The text to improve')
args = parser.parse_args()


llm = ChatOllama(model="llama3.2", base_url='http://localhost:11434',)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an american english expert, and you need to improve the text sent as input."),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print("--- stream from LLama3 via a simple LangChain chain")

for chunks in chain.stream({"input": args.text}):
    print(chunks)
