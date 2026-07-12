# A deep research implementation with Agno

This folder includes code to develop a deep research set of agents, workflow or/and team based on [Agno's documentation](https://docs.agno.com/use-cases/deep-research/overview), [the investment team repository](https://github.com/agno-agi/investment-team/) and other articles: [Research Agent](https://docs-v1.agno.com/examples/agents/research-agent).

This is a step by step to understand how to build this kind of solution.

* [SDK doc](https://docs.agno.com/features/sdk) and [reference](https://docs.agno.com/reference/agents/agent)

## Requirements

* OpenAI compatible LLM - run local (oMLX) as much as possible
* Tool calling, knowledge and RAG
* Terminal command only as of now.

## Individual Agents

The  [investment team repository](https://github.com/agno-agi/investment-team/) has an committee context that is interesting to inject common system prompt. For market studies, a market_analyst, a knowledge_agent, a memo writer agent, and a risk officer agent are relevant and their prompt may be adapted. It also uses 3 layers knowledge with static content, research library and memo archive. It seems to be a very good pattern.

### Market Analyst Agent

* Classic Agent definition with [OpenAILike](https://docs.agno.com/reference/models/openai-like)
* Use [DuckDuckGo](https://docs.agno.com/tools/toolkits/search/duckduckgo) to search web content via DuckDuckGo backend
* [YFinanceTools](https://docs.agno.com/tools/toolkits/others/yfinance) Access to Yahoo finances information. 
* []()

### Understanding YFinance tool

See tests/test_yfinance.py

```sh
uv run python -m pytest deep_researcher/tests/test_yfinance.py -v
uv run python -m pytest deep_researcher/tests/test_yfinance.py -v -s -k assessment_report
```

Example of one of the response:

```json
{
      "method": "get_stock_fundamentals",
      "args": [
        "IBM"
      ],
      # ...
      "parsed": {
        "symbol": "IBM",
        "company_name": "International Business Machines Corporation",
        "sector": "Technology",
        "industry": "Information Technology Services",
        "market_cap": 267716935680,
        "pe_ratio": 21.22083,
        "pb_ratio": 8.119032,
        "dividend_yield": 2.37,
        "eps": 11.31,
        "beta": 0.665,
        "52_week_high": 332.46,
        "52_week_low": 212.34
      }
    },
```

### Understanding DuckDuckGo

To get results use  tools=DuckDuckGoTools(backend="auto"), then use tools.web_search or tools.search_news.

```sh
uv run python -m pytest deep_researcher/tests/test_duckduckgo.py -v -s -k assessment_report
```

### Workflow Step wrapper

When integrating an agent as a step of a workflow the declaration looks like:

```
```

which can be unit tests as in test_workflow.py

```python

```

Example of StepOuput

```python
StepOutput(
    step_name='Market Assessment', 
    step_id='407665e9-57db-4c8c-83ca-8afee611b983', 
    step_type=<StepType.STEP: 'Step'>, 
    executor_type='agent', 
    executor_name='Market Analyst', 
    content="....",
```