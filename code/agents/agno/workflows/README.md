# Workflows running locally

This is based on workflows examples from Agno cookbook.


## 1 daily_ai_news_search_summary

The workflow has 4 steps. Each step generates StepStartedEvent and StepCompletedEvent.

```python
        prepare_input_for_web_search,
        research_team,
        prepare_input_for_writer,
        writer_agent,
```

A step has an step_id. A workflow has a run_id that is passed to steps.

When job is stated, it has a "session_id" that is also passed to steps

The flow has 4 agents, 2 of them are part of a team of researcher.

The workflow object references a database to persist the workflow_session execution. 