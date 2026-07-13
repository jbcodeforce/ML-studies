---
title: "Tree of Thoughts"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/prompt-eng.md]
related: [prompt-engineering, chain-of-thought, prompt-chaining]
tags: [tree-of-thoughts, tot, prompting, reasoning, llm]
---

# Tree of Thoughts

Tree of Thoughts (ToT) is a generalization of Chain-of-Thought prompting, where thoughts represent coherent language sequences that serve as intermediate steps toward solving a problem.

## How It Works

Unlike linear CoT, ToT organizes thoughts in a **tree structure**. At each branching point, the model generates multiple possible reasoning paths. Search algorithms are then used to explore and evaluate the tree:

- **Breadth-First Search (BFS)**: Explores all thoughts at a given depth before moving deeper.
- **Depth-First Search (DFS)**: Explores one branch fully before backtracking.

A multi-round conversation is used to assess the best combination of thoughts and arrive at the optimal solution.

## Comparison to CoT

| Feature | Chain of Thought | Tree of Thoughts |
|---------|-----------------|------------------|
| Structure | Linear sequence | Tree with branches |
| Exploration | Single path | Multiple paths evaluated |
| Search | None | BFS, DFS, or other search algorithms |
| Complexity | Lower | Higher |

## When to Use

ToT is useful for problems that benefit from exploring multiple reasoning strategies, such as complex planning, creative problem-solving, or situations where a single reasoning path may lead to suboptimal results.

## Sources
- [Prompt Engineering Summary](../summaries/prompt-eng.md)

## Related
- [Prompt Engineering](prompt-engineering.md)
- [Chain of Thought](chain-of-thought.md)
- [Prompt Chaining](prompt-chaining.md)