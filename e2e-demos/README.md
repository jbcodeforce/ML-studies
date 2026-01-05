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

## Think deeply

A demonstration of a prompt to use the "Cycle of Critical Thinking", Dr. Justin's a 5-stage thinking framework: Evidence, Assumptions, Perspectives, Alternatives, and Implications, to help getting deeper analysis on a given subject.

How to use it:

* For **Problem-Solving**: Use it on a tough work or personal problem to see it from all angles.
* For **Debating**: Use it to understand your own position and the opposition's so you can have more intelligent discussions.
* For **Studying**: Use it to deconstruct dense topics for an exam. You'll understand it instead of just memorizing it.

The goal isn't to get a quick answer. The goal is to deepen your understanding.