# Taipy

[TapPy](https://docs.taipy.io/) a Python open-source library designed for easy development of data-driven web applications.

It generates web pages from a Flask Server. The main class is `Gui`.

* Support multiple pages
* Keep state of application variables with dynamic binding
* User interactions are event driven
* Pages may be defined by Html or markdown template, or built by code. Page has name for navigation
* Include a CLI to create app or run them.
* Blocks let you organize controls (or blocks) in pages

## Some how to

* Pages are created in different modules, the variables that they can bind to visual elements may have a scope limited to their origin module.
* For Single Page Application we need to associate one page to "/"

## CLI

## Code 

* [1st UI](https://github.com/jbcodeforce/ML-studies/blob/master/techno/taipy/1st_ui.py)
* [Markdown, html, navbar based pages](https://github.com/jbcodeforce/ML-studies/blob/master/techno/taipy/md_ui.py)
* [A chatbot to integrate LangGraph for prompt builder](https://github.com/jbcodeforce/ML-studies/blob/master/llm-langchain/langgraph/chatbot_graph_ui.py)