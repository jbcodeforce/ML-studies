from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os,sys, asyncio

import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")
try:
    api_key=os.environ.get("ANTHROPIC_API_KEY")
except KeyError:
    print(" Set the ANTHROPIC_API_KEY environment variable")
    sys.exit(1)

print("------ Welcome to simple Anthropic Claude text improvement client")


def get_args():
    parser = argparse.ArgumentParser(description='Simple text improvement with Claude ')
    parser.add_argument('text', type=str, help='The text to improve')
    args = parser.parse_args()
    return args.text

def build_chain():
    chat = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")

    system = (
        "You are a helpful technical writer that improve the following content:"
    )
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | chat

async def processMessage(chain, txt):
    for chunk in chain.stream({"text": txt}):
        print(chunk.content, end="", flush=True)
    


if __name__ == "__main__":
    txt = get_args()
    chain = build_chain()
    asyncio.run(processMessage(chain,txt))
    

