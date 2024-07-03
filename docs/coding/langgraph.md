# LangGraph

[LangGraph](https://python.langchain.com/docs/langgraph) is a library for building stateful, **multi-actor** applications, and being able to add cycles to LLM app. It is not a DAG. 

Single and multi-agent flows are described and represented as graphs.

## Value propositions 

* build stateful, multi-actor applications with LLMs
* coordinate multiple chains or actors across multiple steps of computation in a cyclic manner
* build plan of the actions to take
* take the actions
* observe the effects
* support persistence to save state after each step in the graph. This allows human in the loop pattern
* Support Streaming

## [Concepts](https://langchain-ai.github.io/langgraph/concepts/)

[States](https://python.langchain.com/docs/langgraph/#stategraph) may be a collection of messages or custom states as defined by a TypedDict schema. States are passed between nodes of the graph.  [MessageState]() is a predefined states.

`Nodes` represent units of work.  It can be either a function or a runnable. Each node updates this internal state and returns it after execution.

`Graph` defines the organization of the node workflow. Graphs are immutable so are compiled once defined:

```python
graph = MessageGraph()

graph.add_node("chatbot", chatbot_func)
graph.add_edge("chatbot", END)

graph.set_entry_point("chatbot")

runnable = graph.compile()
```

`add_node()` takes an **function or runnable**, with the input to the runnable is the entire current state.

### Agents

Graphs helps implementing Agents as AgentExecutor is a deprecated API. They most likely use tools. The graph development approach is:

1. Define the tools to be used
1. Define the state and what needs to be persisted
1. Define the workflow as a graph, and persistence when needed, compile the graph into a LangChain Runnable. Once the graph is compiled, the application can interact with the graph via stream or invoke methods.

    ```python
    app = workflow.compile(checkpointer=checkpointer)
    ```

1. invoke the graph as part of an API, an integrated ChatBot, ...

Graphs such as StateGraph's naturally can be composed. Creating subgraphs lets you build things like multi-agent teams, where each team can track its own separate state.

LangGraph comes with built-in persistence, allowing developer to save the state of the graph at point and resume from there.

```python
memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
```

See [other checkpointer ways to persist state](https://langchain-ai.github.io/langgraph/reference/checkpoints/#implementations), [AsyncSqliteSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#asyncsqlitesaver) is an asynchronous checkpoint saver that stores checkpoints in a SQLite database or [SqliteSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#sqlitesaver) for synchronous storage is SQLlite..

```python
memory = AsyncSqliteSaver.from_conn_string("checkpoints.sqlite")
```


* See [first basic program](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/langgraph/FirstGraph.py) to call Tavily tool for searching recent information about the weather in San Francisco, it is based on the [tutorial](https://langchain-ai.github.io/langgraph/#example). 

#### Invocation and chat history

The MessageState keeps an array of messages. So the input is a dict with "messages" and then a HumanMessage. As graphs are stateful, it is important to pass a thread_id.

```python
app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
```

The execution trace to LLM presents the following content:

```json
 Entering LLM run with input:
{
  "prompts": [
    "Human: what is the weather in sf"
  ]
}
```

The LLM is generating some statement that tool calling is needed by matching ti the tool named specified during LLM creation.

```json
"generations": [
    [
      {
        "text": "",
        "generation_info": {
          "finish_reason": "tool_calls",
          ...
        "tool_calls": [
            {
            "name": "tavily_search_results_json",
            "args": {
                "query": "weather in San Francisco"
            },
            "id": "call_Vg6JRaaz8d06OXbG5Gv7Ea5J"
            }
```


Graph cycles the steps until there are no more `tool_calls` on AIMessage: 1/ If AIMessage has tool_calls, "tools" node executes, 2/ the "agent" node executes again and returns AIMessage. Execution progresses to the special `END` value and outputs the final state

Adding a "chat memory" to the graph with LangGraph's checkpointer to retain the chat context between interactions.

### Tool Calling

Graph may include `ToolNode` to call function or tool which can be called via conditions on edge. The following declaration uses the predefined langchain tool definition of TavilySearch. The `TavilySearchResults` has function name, argument schema and tool definition so the prompt sent to LLM has information like about the tool like:  "name": "tavily_search_results_json"

```python
from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools)
```

**Conditional edge** between nodes, helps to build more flexible workflow: based on the output of a node, one of several paths may be taken.


## Use cases

The interesting use cases for LangGraph are:

- workflow with cycles and conditional output
- planning agent for plan and execute pattern
- using reflection and self critique
- multi agent collaboration, with or without supervisor
- human in the loop (by adding an "interrupt" before a node is executed.)

### Reasoning and Acting (ReAct) implementation

See [this paper: A simple Python implementation of the ReAct pattern for LLMs](https://til.simonwillison.net/llms/python-react-pattern) from Simon Willison, and the raw code implementation using openAI API [code: ReAct.py](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/langgraph/ReAct.py). LangGraph uses a [prebuilt implementation of ReAct]() that can be tested by []() 
or the [implementation of ReAct using LangGraph](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/langgraph/ReAct_lg.py).

An interesting prompt to use in the ReAct implementation [hwchase17/react](https://smith.langchain.com/hub/hwchase17/react).

## 

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