from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os,sys
import argparse

try:
    api_key=os.environ.get("ANTHROPIC_API_KEY")
except KeyError:
    print(" Set the ANTHROPIC_API_KEY environment variable")
    sys.exit(1)

parser = argparse.ArgumentParser(description='Simple text improvement with Claude ')
parser.add_argument('text', type=str, help='The text to improve')
args = parser.parse_args()

chat = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229")

system = (
    "You are a helpful technical writer that improve the following content:"
)
human = args.text
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

for chunk in chain.stream({}):
    print(chunk.content, end="", flush=True)


