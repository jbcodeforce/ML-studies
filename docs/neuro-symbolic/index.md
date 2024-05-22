# Neuro Symbolic AI

Neuro Symbolic AI combines elements of neuron networks and symbolic reasoning to develop intelligent systems. Neural networks excel at pattern recognition and learning from data, while  traditional AI like rule engines focuses on logic, reasoning, and symbolic representations.

The goal is to create more interpretable, explainable, and robust AI systems.

Symbolic reasoning can operate on the structured knowledge graph to perform tasks such as question answering, knowledge inference, and logical reasoning.

Knowledge graphs represent information in a structured and semantically rich manner, using entities, relationships, and attributes

LLMs are trained on static document sets, which means gaps exist with newly created knowledge. RAG helps to address this problem, but there is still gaps in semantic controlled response.

## Use Cases

We may consider three entry points for transforming existing business processes with AI and automation: 

1. Process automation using STP, integration of human workflow with document classification and data capture.
1. Decision based on business policies, including risk scoring, fraud detection, KYC
1. Improved user experience to find solutions and make adhoc decisions.

LLM are amazing tool for understanding and generating natural language, however they are not able to make consistent business decisions. 

* **Healthcare**: Deep learning can do medical images or analog graph pattern recognition, predictive analytics, classification, with symbolic reasoning to deliver personalized treatment recommendations, or help on diagnostic.
* **Complaint management:** combine workflow, chatbot, decision rules for next best actions, product recommendation, ML for sentiment analysis. 
* **Financial risk management**: is about continuously and dynamically altering the user experience to reduce false positives and slow down or stop adversarial patterns. Risk identification is about integrating event and real time processing to get user action context within time windows. Risk context comes from data that may come from different sources. Risk scoring includes a combination of ML model developed on top of structured or unstructured data (anomaly detection), and business logic coded in a form of `if condition then action` rules and rule flow to organize the rule execution in minimum calls. Decision acts in the moment of the transaction.

???- example "Some risky behaviors"
        1. If user tried to login 3 times in the last 10 minutes, change the password, add a new beneficiary, and trigger a transfer once the account is accepted. Some data elements may be of interest to assess for potential fraud, like the user IP address, the country of origin, the type of bank or country for the beneficiary.
        1. Two gas transactions within 5 minutes in the same gaz stations is most likely a fraud.
        1. Transaction amount higher than average spent habit.

As soon as we add a touch point with a human, like a natural language interface, we have to think about risk and playing the system.

## Intelligent Assistant

A tool which accesses the business applications any user accesses during a work day, gathers the information, curates it. It interacts with natural language, understands intent, completes a multi-step tasks across applications, systems, and people. The assistant learns over time on how we work with the systems.

WatsonX Orchestrate uses NLP, Gen AI and skills to help implement custom orchestration. Skill is function wrapper with description. IBM predefined a set of skill like integration to SAP, Gmail...

As any API can be wrapped into a skill or tool, then it can be orchestrated by a LLM. 

## Semantic Router

A [semantic router](https://github.com/aurelio-labs/semantic-router) serves as a sophisticated decision-making layer that can select the most appropriate language model or response for each user query. It uses semantic vector space to align user questions with the most fitting predefined responses. The router helps LLM or any action to make decisions or augment query with more info or add more context to find the best answers from a list of possible ones.

Some use cases where semantic routing will be relevant:

* Defense against malicious query attacks as it may discern and counteract potential threats.
* Avoid sensitive topic to avoid inappropriate content
* Simplify function calling within applications
* Optimize database query and RAG query

## Sources

* [Solving Reasoning Problems with LLMs in 2023](https://towardsdatascience.com/solving-reasoning-problems-with-llms-in-2023-6643bdfd606d)
* [Connecting AI to Decisions with the Palantir Ontology](https://blog.palantir.com/connecting-ai-to-decisions-with-the-palantir-ontology-c73f7b0a1a72?gi=b4f8020a603a)
* [Semantic Router super fast decision layer for LLMs and AI agents.](https://www.geeky-gadgets.com/semantic-router-superfast-decision-layer-for-llms-and-ai-agents/)