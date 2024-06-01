# Agentic AI

[Agent](https://lilianweng.github.io/posts/2023-06-23-agent/) is an orchestrator pattern where the LLM decides what actions to take from the current query and context. With chain, developer code the sequence of tasks, with agent the LLM decides. 

The reference architecture for an agent looks like what Lilian Weng illustrated in the following figure (light adaptation):

![](./diagrams/agent-ref-arch.drawio.png)

The planning phase includes techniques like Chain of Thought ("think step by step"), Tree of thoughts (explores multiple reasoning paths) or LLM+P (used external long-horizon planner).

Short term memory is the context, and limited by the LLM context window size. Long term memory is the vector store supporting the maximum inner product search, it is also used to self improve agents. Entity memory is a third type of memory to keep information of the subjects of the interactions or work to be done. Short term memory helps exchanging data between agents too. 

Tools are used to call external services or other LLMs. Neuro-symbolic architecture can be built with expert system modules combined with general-purpose LLM. LLM routes to the best tool.

There are [different types](https://python.langchain.com/docs/modules/agents/agent_types/) of agent: Intended Model, Supports Chat, Supports Multi-Input Tools, Supports Parallel Function Calling, Required Model Params.


## Guidelines

Agents perform better if we define a role to play, instruct them with prompt to help them to focus on a goal, add tools to access external systems, combine them with other agents to cooperate.  and chain content between agents. 

Focus is becoming important as the context windows are becoming larger. With too many information LLM can lose the important points and goals. Try to think about multiple agents to do better work together.

Too much tools adds confusion for the agents, as they have hard time to select tool, or distinguish what is a tool, a context or an history. Be sure to give them tools for what they need to do. 

For task definition, think about process, actors and tasks. Have a clear definition for each task, with expectation and context. Task may use tools, should be able to run asynchronously, output in different format like json, xml, ...

## Use cases

* Agents to plan an article, write this article and review for better edition. [research-agent.py](https://github.com/jbcodeforce/ML-studies/blob/master/techno/crew-ai/research-agent.py)
* Support Representative, the [support_crew.py](https://github.com/jbcodeforce/ML-studies/tree/master/techno/crew-ai/support_crew.py) app demonstrates two agents working together to address customer's inquiry the best possible way, using some sort of quality assurance. It uses memory and web scrapping tools.
* Customer outreach campaign: [customer_outreach.py](https://github.com/jbcodeforce/ML-studies/tree/master/techno/crew-ai/customer_outreach.py) uses tools to do google searches with two agents doing lead analysis.
* Crew to tailor job application with multiple agents [job_application.py](https://github.com/jbcodeforce/ML-studies/tree/master/techno/crew-ai/job_application.py)

## Design Patterns

## CrewAI

[crewAI](https://www.crewai.com/) is a framework to develop application using multiple-agent. It uses the concepts of Agent, Task and Crew to organize the work between agents. The concepts are common to any Agentic AI solutions.

```python
from crewai import Agent, Task, Crew
```

Agent needs the following 6 elements:

1. Role Playing: Agents perform better when doing role playing. It is mapped to the first statement in a prompt, and it is a common practice in prompt engineering.

    ```python
    writer = Agent(
                role="Content Writer",
                goal="Write insightful and factually accurate "
    ```

1. Focus on goals and expectations to better prompt the agent: "give me an analysis of xxxx stock". Too much stuff in the context window is confusing the model, and may hallucinate. May be splitting into multiple agents is a better solution instead of using a single prompt.

1. Tool is used to call external system, and is well described so the model can build parameters for the function and be able to assess when to call the function. Now too many tools will also add to the confusion. Small model will have hard time to select tools. So think to have multiple-agent with only the tools they need to do their task.
1. Cooperation has proved to deliver better results than unique big model. Model can take feedbacks from each others, they can delegate tasks.
1. Guardrails are helping to avoid models to loop over tool usages, creating hallucinations, and deliver consistent results. Models work on fuzzy input, generate fuzzy output, so it is important to be able to set guardrails to control outcomes or runtime execution.
1. Memory is important to keep better context, understand what was done so far, apply this knowledge for future execution. Short term memory is used during the crew execution of a task. It is shared between agents even before task completion. Long term memory is used after task execution, and can be used in any future tasks. LTM is stored in a DB. Agent can learn from previous executions. This should lead agent to self-improve. The last type of memory is the entity memory (person, organization, location). It is also a short term, and keep information of the entity extracted from NLP.

CrewAI has tools to scrape website, search internet ([Serper](https://serper.dev/)), load customer data, tap into previous conversations, load data from a CRM, checking existing bug reports, checking existing feature requests, checking ongoing tickets...

See code examples in [the techno/crew-ai folder](https://github.com/jbcodeforce/ML-studies/tree/master/techno/crew-ai)

### Some guidelines

* Adapt the task and agent granularity
* Task can be executed in different ways, parallel, sequential,... so test and iterate
* With agent delegation parameter, the agent can delegate its work to another agent which is better suited to do a particular task.
* Try to add a QA agent to control and review task results
* Tools can be defined at the agent level so it will apply to any task, or at the task level so the tool is only applied at this task. Task tool overrides agent tools.
* Tools need to be versatile, fault tolerant, and implement caching. Versatile to be able to get the fuzzy input well interpreted by the model and call the relevant tools and by extracting structured input parameters in the form of json or key-value pairs. 
* To be fault tolerant, function can stop execution, retries with exponential backoff, or report error message to the LLM so it can better extract and format parameters.
* CrewAI offers a cross-agent caching mechanism. It is also compatible with LangChain tools.
* Think as a manager: define the goal and what is the process to follow. What are the people I need to hire to get the job done. Use keyword and specific attributes for the role, agent needs to play.

## AutoGen

[Microsoft AutoGen](https://microsoft.github.io/autogen/) is a multi-agent conversation framework to help developers build LLM workflows. The first abstraction is a ConversableAgent

## References

* [LLM Powered Autonomous Agents - Lilian Wang](https://lilianweng.github.io/posts/2023-06-23-agent/)
* [Prompt engineering with external APIs](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/#external-apis)
* [Crew-ai tutorial on deeplearning.ai](https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai)