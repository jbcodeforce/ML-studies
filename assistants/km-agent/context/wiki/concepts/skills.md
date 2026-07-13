---
title: "Skills"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/review.md]
related: [agentic-ai, agentic-design-patterns, tool-calling]
tags: [skills, agent, packaging, SKILL.md, agent-capabilities]
---

# Skills

Skills package reusable agent capabilities (prompts, tools, workflows) for AI coding assistants and orchestration platforms. They enable agents to discover, load, and execute specialized functionality on demand.

## Structure

A standard skill is packaged as a self-contained folder:

```
my-specialized-skill/
├── SKILL.md          # Core specification, triggers, & instructions
├── scripts/          # Deterministic executable scripts (Python, Bash, etc.)
└── templates/        # Boilerplate files, assets, or reference docs
```

## SKILL.md Format

The `SKILL.md` file has two key parts:

1. **YAML Frontmatter (Metadata):** A crisp `name` and hyper-focused `description`. The description acts as a trigger condition — treating it like regex for the agent's brain. If too broad, the agent triggers it mistakenly; if too narrow, the agent won't reuse it.

2. **Procedural Body:** Standard Markdown containing multi-step workflows, conditional logic, and specific tool execution expectations.

## Execution Best Practices

- **Deterministic Scripts:** Steps requiring zero improvisation (e.g., parsing CSV, calling APIs) should be scripted rather than written as natural language instructions.
- **Prevent Plan Drift:** Use clear, sequential step boundaries to force predictable execution: *Gather context → Take action → Verify results*.
- **State Safety:** Ensure skill scripts don't cause state collisions when invoked by parallel sub-agents.

## Validation & Evaluation

- **Linting:** Ensure YAML frontmatter fields are complete and structure complies with standard formats.
- **Paired Simulation:** Evaluate agent trajectories *with* the skill versus a baseline *without* it to measure efficiency, accuracy, and safety improvements.

## Related Research

- "Skills Are the New Apps– Now It's Time for Skill OS" — Le Chen and co.
- Agentic Continuous Evaluation of Skills (ACES) — Kevin C.
- SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization

## Sources
- [AI Discussions Review](../summaries/review.md)

## Related
- [Agentic AI](agentic-ai.md)
- [Agentic Design Patterns](agentic-design-patterns.md)
- [Tool Calling in LLMs](tool-calling.md)