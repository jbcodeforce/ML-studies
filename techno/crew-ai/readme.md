# Crew AI study

Content on [crewai](https://www.crewai.com/) based on Deeplearning.AI training and product documentation.

CrewAI uses the concepts of Agent, Task and crew to organize the work between agents.

It also has tools to scrape website, search internet, load customer data
Tap into previous conversations, load data from a CRM, checking existing bug reports, checking existing feature requests, checking ongoing tickets.

## Some guidelines

* Agents perform better when doing role playing
* Focus on goals and expectations ot better prompt the agents
* Adapt the task and agent granularity
* Task can be executed in different ways, parallel, sequential,... so test and iterate
* Long term memory is used after task execution, and can be used in any future tasks. 
* With agent delegation parameter, the agent can delegate its work to another agent which is better suited to do a particular task.
* Try to add a QA agent to control and review task results
* Tools can be defined at the agent level so it will apply to any task, or at the task level so the tool is only applied at this task. Task tool overrides agent tools.
* Think as a manager: define the goal and what is the process to follow. What are the people I need to hire to get the job done. Use keyword and specific attributes for the role, agent needs to play.

## Code

### Agent to do research

The code [research-agent.py](https://github.com/jbcodeforce/ML-studies/tree/master/techno/crew-ai/research-agent.py) illustrates the use of one agent to plan for the documentation, then a writer agent taking the plan to write a short blog and last an editor agent to fix the content for compliance to internal writing rules.

### Support Representative

The [support_crew.py](https://github.com/jbcodeforce/ML-studies/tree/master/techno/crew-ai/support_crew.py) app demonstrates two agents working together to address customer's inquiry the best possible way, using some sort of quality assurance. It uses memory and web scrapping tools. 
