from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import requests, json

OLLAMA_URL="http://localhost:11434/api/generate"

def chat_using_langchain(message: str):
    llm = ChatOllama(model="mistral")
    prompt = ChatPromptTemplate.from_template("You are a researcher, an expert in AI company.")

    chain = prompt | llm | StrOutputParser()
    topic={"topic": message}

    print(f"send request {topic} to Mistral on Ollama....")
    print(chain.invoke(topic))


def chat_ollama_api(messages):
    r = requests.post(
        OLLAMA_URL,
        json={"model": "mistral", "prompt": messages},
	    stream=False
    )
    r.raise_for_status()
    output = ""
    message = {}
    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("response", "")
            #content = message["content"]
            output += message
            # the response streams one token at a time, print that as we receive it
            print(output, end="", flush=True)

        if body.get("done", False):
            return output

if __name__ == "__main__":
    #chat_using_langchain("do you know Athena Decision Systems company?")
    chat_ollama_api("do you know Athena Decision Systems company?")
    