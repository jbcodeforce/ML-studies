# LangGraph

[LangGraph](https://python.langchain.com/docs/langgraph) is a library for building stateful, **multi-actor** applications, and being able to add cycles to LLM app. It is not a DAG. 

Single and multi-agent flows are described and represented as graphs.

## Value propositions 

* build stateful, multi-actor applications with LLMs
* coordinate multiple chains or actors across multiple steps of computation in a cyclic manner
* build plan of the actions to take
* take the actions
* observe the effects
* support persistence to allow human in the loop pattern

## [Concepts](https://langchain-ai.github.io/langgraph/concepts/)

[States](https://python.langchain.com/docs/langgraph/#stategraph) may be a collection of messages or custom states as defined by a TypedDict schema. States are passed between nodes of the graph.  Nodes represent units of work.  It can be either a function or a runnable. Each node updates this internal state with its return value after it executes.

Graph definitions are immutable so are compiled once defined:

```python
graph = MessageGraph()

graph.add_node("chatbot", chatbot_func)
graph.add_edge("chatbot", END)

graph.set_entry_point("chatbot")

runnable = graph.compile()
```

`add_node()` takes an **function or runnable**, with the input to the runnable is the entire current state.

Graph may include `ToolNode` to call function or tool which can be called via conditions on edge. Conditional edge helps to build more flexible workflow: based on the output of a node, one of several paths may be taken.

Once the graph is compiled, the application can interact with the graph via stream or invoke methods.

LangGraph comes with built-in persistence, allowing developer to save the state of the graph at point and resume from there.

```python
memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
```

Graphs such as StateGraph's naturally can be composed. Creating subgraphs lets you build things like multi-agent teams, where each team can track its own separate state.

See [other checkpointer ways to persist state](https://langchain-ai.github.io/langgraph/reference/checkpoints/#implementations), [AsyncSqliteSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#asyncsqlitesaver) is an asynchronous checkpoint saver that stores checkpoints in a SQLite database or [SqliteSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#sqlitesaver) for synchronous storage is SQLlite..

```python
memory = AsyncSqliteSaver.from_conn_string("checkpoints.sqlite")
```

## Use cases

The interesting use cases for LangGraph are:

- workflow with cycles and conditional output
- planning agent for plan and execute pattern
- using reflection and self critique
- multi agent collaboration, with or without supervisor
- human in the loop (by adding an "interrupt" before a node is executed.)

### Reason Act (ReAct) implementation

See [this paper: A simple Python implementation of the ReAct pattern for LLMs](https://til.simonwillison.net/llms/python-react-pattern) from Simon Willison, and a raw code using openAI API [code: ReAct.py](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/langgraph/ReAct.py)
and the [one using LangGraph](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/langgraph/ReAct_lg.py)

An interesting prompt to use in the ReAct implementation [hwchase17/react](https://smith.langchain.com/hub/hwchase17/react).

### 

## Code 

See [code samples](https://github.com/langchain-ai/langgraph/tree/main/examples) in my [own sample folder](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/langgraph). 

See the [owl agent framework open source project](https://athenadecisionsystems.github.io/athena-owl-core/) to manage assistant, agents, tools, prompts..

## Code FAQ

???- question "prompt variables to be integrated in LangGraph"
        

## Deeper dive

* [LangGraph product reference documentation.](https://langchain-ai.github.io/langgraph/reference/prebuilt/)
* [LangGraph git repository](https://github.com/langchain-ai/langgraph)
* [LangGraph API reference guide](https://langchain-ai.github.io/langgraph/reference/graphs/)
* [Deeplearning.ai AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) with matching code 
* [A simple Python implementation of the ReAct pattern for LLMs](https://til.simonwillison.net/llms/python-react-pattern)