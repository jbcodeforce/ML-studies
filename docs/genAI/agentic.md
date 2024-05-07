# Agentic AI

[Agent](https://lilianweng.github.io/posts/2023-06-23-agent/) is an orchestrator pattern where the LLM decides what actions to take from the current query and context. With chain, developer code the sequence of tasks, with agent the LLM decides. 

The reference architecture for an agent looks like what Lilian Weng illustrated in the figure (light adaptation):

![](./diagrams/agent-ref-arch.drawio.png)

The planning phase includes techniques lie Chain of Thought ("think step by step"), Tree of thoughts (explores multiple reasoning paths) or LLM+P (used external long-horizon planner).

Short term memory is the context, and limited by the LLM context window size. Long term memory is the vector store supporting the maximum inner product search.

Tools are used to call external services or other LLMs. Neuro symbolic architecture can be built with expert system modules combined with general-purpose LLM. LLM routes to the best tool.

There are [different types](https://python.langchain.com/docs/modules/agents/agent_types/) of agent: Intended Model, Supports Chat, Supports Multi-Input Tools, Supports Parallel Function Calling, Required Model Params.



## References

* [LLM Powered Autonomous Agents - Lilian Wang](https://lilianweng.github.io/posts/2023-06-23-agent/)
* [Prompt engineering with external APIs](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/#external-apis)