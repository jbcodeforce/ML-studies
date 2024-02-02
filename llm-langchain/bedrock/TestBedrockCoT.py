import json
import os
import sys, getopt

from langchain_community.llms import Bedrock

module_path = "."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

bedrock_runtime = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

claude_inference_modifier = {
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

default_instance_modifier= {
    "temperature": 0.7,
    "maxTokens":300,
}

def textPlayground(model,prompt):
    if "claude" in model:
        inference_modifier= claude_inference_modifier
    else:
        inference_modifier=default_instance_modifier
    textgen_llm = Bedrock(
        model_id=model,
        client=bedrock_runtime,
        model_kwargs=inference_modifier,
    )
    response = textgen_llm.invoke(prompt)
    print_ww(response)
    return response

def usage():
    print("Usage: python TestBedrockCoT.py [-h | --help] [ -p file_name_prompt -m model_name ]")
    print("Example: python TestBedrockCoT.py -p cot1.txt -m anthropic.claude-v2")
    sys.exit(1)

def processArguments():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:p:m:", ["help","file_name_prompt","model_name"])
    except getopt.GetoptError as err:
        usage()

    FILENAME="cot1.txt"
    MODEL_NAME="anthropic.claude-v2"
    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            usage()
        elif opt in ["-p", "--file_name_prompt"]:
            FILENAME = arg
        elif opt in ["-m", "--model_name"]:
            MODEL_NAME = arg
    return FILENAME,MODEL_NAME

def buildPrompt(FILENAME):
    p=""
    with open(FILENAME) as f:
        for line in f:
            p= p + line
    return p

if __name__ == '__main__':
    FILENAME,MODEL_NAME=processArguments()
    prompt=buildPrompt(FILENAME)
    print(prompt)
    textPlayground(MODEL_NAME,prompt)