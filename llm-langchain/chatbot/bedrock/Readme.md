# A python app for a chatbot using Bedrock LLM

A chatbot with LLM and contextual aware embeddings. This implementation use LangChain library to chain the components. See my [LangChain deeper dive here](https://jbcodeforce.github.io/ML-studies/coding/langchain/) with a lot of code examples using Bedrock in [this folder](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain).

## Start virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Can also use the docker file and then the image from [ML-studies repository](https://github.com/jbcodeforce/ML-studies).

Once the python environment use the setup.sh script to download the last SDK version to get access to bedrock APIs.