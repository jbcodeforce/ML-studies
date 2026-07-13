---
title: "Small Specialist Agents"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/agentic.md]
related: [agentic-ai, agentic-design-patterns, langgraph]
tags: [agentic, small-specialist-agents, ooda, hierarchical-planning, multi-agent, openssa]
---

# Small Specialist Agents

Small Specialist Agents (SSAs) is an agentic approach to perform planning and reasoning to enhance AI capabilities for complex problem-solving using domain-specific knowledge. Rather than relying on a single large agent with broad capabilities, SSAs use multiple focused agents with specialized knowledge.

## OODA Loop

SSAs may implement the OODA loop: Observe, Orient, Decide, and Act. This creates a continuous feedback cycle where agents:

- **Observe**: Gather information from the environment
- **Orient**: Contextualize and interpret observations
- **Decide**: Choose actions based on domain knowledge
- **Act**: Execute decisions and feed results back

## Hierarchical Task Planning

SSAs cut bigger tasks into smaller ones through hierarchical task planning. Planning can use up-to-date data to define future actions. This enables agentic AI to respond swiftly and effectively to changing environments.

## Applications

- **Predictive maintenance**: Predict maintenance needs, adjust operational parameters to prevent downtime
- **Energy optimization**: Ensure energy production meets demand without excess waste
- **Healthcare**: Analyze genetic data, medical histories, and real-time responses to various treatments

## Advantages Over Large Agents

- Each agent has a focused prompt with a smaller, more manageable tool list
- Reduced confusion in tool selection (avoiding the Multi-Action-Agent problem)
- Enables use of smaller, cheaper LLMs per agent
- More deterministic results with local code and MCP servers
- Better control over agent behavior and outcomes

## Trade-offs

The multi-agent approach adds complexity in designing, implementing, and tuning the solution. However, it authorizes the usage of smaller LLMs, local code/MCP servers, and specific prompts.

## See Also

- [OpenSSA project](https://github.com/aitomatic/openssa)

## Sources
- [Agentic AI](../summaries/agentic.md)

## Related
- [Agentic AI](agentic-ai.md)
- [Agentic Design Patterns](agentic-design-patterns.md)
- [LangGraph](langgraph.md)