# Taipy

[TapPy](https://docs.taipy.io/) a Python open-source library designed for easy development of data-driven web applications. It supports defining scenario for data pipeline and integrate with UI elements to do the data presentations and interactions.

It generates web pages from a Flask Server. The main class is `Gui`.

* Support multiple pages which are defined in markdown, html or python
* Offer various visual elements that can interact with the Python variables and environment
* Every callback, including submit_scenario(), receives a State object as its first parameter. 
* Keep State of user connection and variables for dynamic binding.
* User interactions are event driven
* Page has name for navigation
* Include a CLI to create app or run them.
* Blocks let you organize controls (or blocks) in pages
* Scenarios are global variables available to everyone connected.

## Some how to

* Pages are created in different modules, the variables that they can bind to visual elements may have a scope limited to their origin module.
* For Single Page Application we need to associate one page to "/"

## CLI

'''sh
taipy run main.py
'''

## Code 

* [1st UI](https://github.com/jbcodeforce/ML-studies/blob/master/techno/taipy/1st_ui.py)
* [Markdown, html, navbar based pages](https://github.com/jbcodeforce/ML-studies/blob/master/techno/taipy/md_ui.py)
* [A chatbot to integrate LangGraph for prompt builder](https://github.com/jbcodeforce/ML-studies/blob/master/llm-langchain/langgraph/chatbot_graph_ui.py)