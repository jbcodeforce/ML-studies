# Summary: Agentic AI

**Source:** `raw/studies/genAI/agentic.md`

## Main Thesis
Agentic AI is an orchestrator pattern where LLMs decide what actions to take from the current query and context, following Lilian Weng's reference architecture. The key insight is that agent results quality is only 20–30% linked to the LLM model — the remaining 80% depends on scaffolding quality, including tools, prompts, context management, and system design.

## Key Architectural Components
- **Planning**: Chain of Thought, Tree of Thoughts, LLM+P for external planners
- **Memory**: Three types — short-term (context window), long-term (vector store), and entity memory (subjects of interactions)
- **Tools**: External service calls, CLI interfaces, MCP servers — each tool should be a single composable capability
- **Neuro-symbolic**: Expert system modules combined with general-purpose LLMs

## Design Patterns Covered
- **Reflect**: Self-critique loop for iterative improvement
- **Router**: LLM-classified query routing to specialized handlers
- **Human-in-the-Loop**: Interrupts for human approval before sensitive operations
- **ReAct**: Alternating reasoning and action cycles
- **Multi-Agent Collaboration**: Sequential and hierarchical agent teams
- **Wiki LLM**: Active librarian pattern for compiled, cross-referenced knowledge

## Frameworks Compared
Agno (best overall), LangGraph (stateful workflows), CrewAI (role-based), Pydantic AI (type-safe), OpenAI SDK (direct control), AutoGen (conversational). After 2 years of experience, the author prefers pure Python over black-box libraries like LangChain.

## Key Challenges
- High cost of open-loop agents
- Instability with new model releases
- Poor tool selection with many tools
- "In-the-middle" context window problem
- Current implementations not yet production-ready for all use cases

## Recommendations
- Use multiple small specialist agents with focused prompts and limited tools
- Code deterministic logic first, integrate as tools later
- Template prompts consistently; design for production with logging, monitoring, and versioning
- Consider event-driven agents triggered by systems via Flink's streaming capabilities