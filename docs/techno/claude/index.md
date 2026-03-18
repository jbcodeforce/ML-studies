# Claude Code

???- info "Updates"
    03/17/2026

An agentic coding assistant which not only address coding assistant, but also discovery and design.

The architecture of the assistant looks like:

![](./images/arch.drawio.png)

* **CLAUDE.md** file represents memory across sessions. It can include style guidelines and common commands. This file is automatically loaded into the context.
* [See overview](https://code.claude.com/docs/en/overview).


## Capabilities

* Conversation history is stored locally, which is loaded into context.
* Can launch sub-agents
* Manage to dos
* Refer back to previous information
* Agentic search

## Integrating with existing git repo.

Here are 3-step playbook for getting Claude Code productive and safe on existing repos.

* Create a first CLAUDE.md file. Don't use /init as-is — it produces bloated output. As good practices:

| Include | Exclude |
| --- | --- |
| Shell commands LLM can't guess | Anything LLM can figure out from code |
| Code style rules that differ from defaults | Standard language conventions |
| Testing instructions and preferred runners | Detailed API docs (links to existing content instead) |
| Architectural decisions specific to your project | Long explanations or tutorials
| Common gotchas or non-obvious behaviors | File-by-file codebase descriptions |

* Adopt TDD
* Plan First, then Scale with skills and plugins.
    * Planning is most valuable when you're uncertain about the approach or the change spans multiple files. Ctrl-G to update the plan.
    * Prefer CLIs over MCP
    * Leverage plugin like [Superpowers](https://claude.com/plugins/superpowers)
    * Manage context aggressively — Use /clear between unrelated tasks, /compact to summarize mid-session, and subagents for investigation

## Useful prompts

| Goal | Prompt |
| --- | ---| 
| Fix build issue | The build fails with this error: [paste error]. Fix it and verify the build succeeds. Address the root cause, don't suppress the error.|
| Fix runtime issue | Use subagents to investigate why [service] is throwing [error]. Check logs and recent changes. |
| Code review | Read ... folder, What are the key files and flows?| 
| Build new feature | I want to build [brief description]. Interview me about edge cases, tradeoffs, and implementation using AskUserQuestion. Then write a spec to SPEC.md |
| Review PR (collaborate on PR) | Review the PR at [link]. Look for edge cases, race conditions, and consistency with existing patterns |
| Address PR feedback | Read the review comments on PR #[number]. Address each one, reply to the threads, and push fixes.|

## Sources

* [Deeplearning training](https://learn.deeplearning.ai/courses/claude-code-a-highly-agentic-coding-assistant)
* [Using Claude for a lot of things]()
* [Use LiteLLM as proxy for Claude code - tutorial](https://docs.litellm.ai/docs/tutorials/claude_responses_api)