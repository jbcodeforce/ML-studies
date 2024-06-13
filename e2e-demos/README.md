# A set of end to end demonstrations

This folder includes a set of end to end demonstrations.

## Pre-requisites

Start virtual environment, and be sure to install specifics libraries in each project with 

```sh
pip install -r requirements.txt
```

Be sure to have a .env file to define API keys under the ML_studies folder.

## QA Retrieval

A Streamlit app to demonstrate the impact of RAG on response quality. It uses the https://lilianweng.github.io/posts/2023-06-23-agent/ as a source for RAG

```sh
streamlit run Main.py  
```

[http://localhost:8501/]()http://localhost:8501/