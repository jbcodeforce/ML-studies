# Agentic AI

[Agent](https://lilianweng.github.io/posts/2023-06-23-agent/) is an orchestrator pattern where the LLM decides what actions to take from the current query and context. With chain, developer code the sequence of tasks, with agent the LLM decides. 

The reference architecture for an agent looks like what Lilian Weng illustrated in the figure (light adaptation):

![](./diagrams/agent-ref-arch.drawio.png)


There are [different types](https://python.langchain.com/docs/modules/agents/agent_types/) of agent: Intended Model, Supports Chat, Supports Multi-Input Tools, Supports Parallel Function Calling, Required Model Params.