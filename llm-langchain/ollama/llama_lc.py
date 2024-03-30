from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import argparse

print(" The LLM is an earth environment expert so ask question about environment.")
parser = argparse.ArgumentParser(description='Simple text improvement with Llama2 ')
parser.add_argument('text', type=str, help='The text to improve')
args = parser.parse_args()


llm = Ollama(model="llama2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a earth environment expert."),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print("--- call LLama2 via a simple LangChain chain")

resp=chain.invoke({"input": args.text})
# The results should includes some hallucinations
print(resp)