# Implementation solution methodology

The adoption of LLM in an enterprise ecosystem should not be an afterthought process. Leaders should go over the hype, and really address what those technologies can help with. Interesting to see that companies who offer hosted LLMs in the cloud are not encouraging their own employees to use those LLMs internally. With text generation and text enhancement, all employees will bring the same quality of results and it will be difficult to measure them.

Currently AI systems are helping to take million of business decisions every days using classical expert systems, or business rule management systems, combined or not with ML predictive scoring. Any business process activities involving human task, may be able to leverage better automation by adopting AI agent to help them doing their task quicker with better results and accuracy. Actually LLM has poor accuracy result to be trustable for any business. But there is way to improve this.

The classical LLM use cases of summarization, text improvement, and idea generations are useful for a lot of busness activities, and halucinations have no dramatic impact as the human is still taking the lead on acting on what the LLM results are. He can judge the result content, pick and choose from the returned text what makes sense or not.

The current tendency to imagine LLM being an orchestrator of functions or services and therefore embbedable into a business process, may not be acceptable in many use cases. Process automation needs consistency, high level of accuracy and coded business logic to manage orchestration and compensation flow when exceptions occur. For each function or service to integrate. developer needs to revieew the interface characteristics which provided semantic and behavior description of the system to integrate with. There are a lot of best practices inherited from SOA, that should help us deciding how much a LLMs can do for integration, and how much they may help human takes the next best actions.

## Introduction

The adoption of generative AI in the enterprise, follows the same project management practices as the ones of any data, analytics and decision system solution implementation. In this chapter, we define what are the specifics requirements, the methodology needs to address and the proposed phase and team engagements to put in place.

The proposed methodology uses the successful practices used during thousands of consulting engagements around decision automation solution implementations.

Always starts by the business problem to solve.

## Classical business process Generative AI can help to address

In the context of a business process, like in insurance claim processing, there are some activities that were already automated with the adoption of decision services, which uses inference rule engine, (the AI of the 90s), with hundred of rules to address insurance product configuration, feature eligibility, pricing computation, basic claim acceptance, risk scoring, to name a few.

Parts of those process, have human activities, and the complete orchestration is done by business process management system or application logic mostly coded as state machines. When humans are involved, they verify documents, correlate information between different data sources, some time rekey data into different systems, and apply some non-coded business policies, but documented somewhere to decide to accept, reject the claim or most likely ask for more information. 

Obviously with SOA and microservice adoption, rekeying data should be history, and most middleware integrations, can move those data between systems. Document processing is already well cocvered by OCR product. The document processing with OCR give good results but are not 100% accurate, Robot Process Automation (RPA) has helped to integrate new data source to legacy systems. 

So how a Gen AI, multi model deep learning, may help in this claim example? Human can get document summarization, query a corporate document corpus to get better semantic search results, extract specific facts relevant to the claim from unstructured data like emails, paper mail scanned in pdf, or even images. 

We can then generalize these to a lot of business processes as soon as they have those characteristics:

- humans are involved to make the process progressing
- the process is running for a long time. Straight throught processing flows are already fine tuned for automation and bringing unstructured response may impact the latency then improve it.
- humans needs a lot of knowledge about the business, the process, some previous cases, to take quick decsion.
- SMEs are leaving company, new comers need to be efficiently trained,
- human has a way to ask to line if management or subject matter experts when a case is more complex, to get advice.
- human may needs to consult other datasources to get better data to take decision. 

## Methodology requirements

The methodology needs to support:

* simple and short requirements discovery phase,
* focuses on the task within a business process where AI may be able to help, so we can narrow the scope to bring immediate value
* the definition of a sustainable architecture, selecting the products or software components for what they are supposed to help with,
* the implementation od  minimum valuable product to proof the value of the proposed solution with working software
* the integration of continuous end user feedbacks with playback the solution

## Best practices

As we focus in business problem to address. the methodology adopts the following well known and adopted practices of:

* **Design thinking** to address human's needs and perspectives of the end-users. It encourages developing empathy map, pain and challenges, end user's problem, develop innovative ideas to solve the problem at hand, and prototyping experimentations to get earlier feedbacks.
* **Lean startup** with Minimum Viable Product implementations, measure against hypothesis and pivot practices.
* **Agile** development practices with test-driven development.
* **GitOps** to simplify hybrid cloud solution deployment, using infrastructure as code
* **Decisions and data focus** to address the need for data, and normalize the decisions to be taken in the context of a business process, a business applications and the data to use.

AI helps to take business decision, if not, it is useless. So understanding the decisiosn to take are keys to the success of any AI project implementation.

### Interface Characteristics

This section lists the full set of interface characteristics to consider for each system to integrate with:

* FUNCTIONAL DEFINITION

    * Principal data objects
    * Operation/function
    * Read or change
    * Request/response objects

* TECHNICAL INTERFACE

    * Transport
    * Protocol
    * Data format

* INTERACTION TYPE

    * Request-response or fire-forget
    * Thread-blocking or asynchronous
    * Batch or individual
    * Message size

* PERFORMANCE

    * Response times
    * Throughput
    * Volumes
    * Concurrency

* INTEGRITY

    * Validation
    * Transactionality
    * Statefulness
    * Event sequence
    * Idempotence

* SECURITY

    * Identity/authentication
    * Authorization
    * Data ownership
    * Privacy

* RELIABILITY

    * Availability
    * Delivery assurance

* ERROR HANDLING

    * Error management capabilities
    * Known exception conditions
    * Unexpected error presentation

From these characteristics, it does not seem realist to design a solution where the LLM service provider will be able to execute some function code from their SaaS platform to integrate on-premises services, with interface characteristic that is no REST, json, stateless, idempotent, synchronous. So outside of toy solution we need something else to get real adoption.

### Generative AI context

Some cloud provider leaders think, that enterprise will have a large library of dedicated models. Training model cost a lot of money for ensure results. Fine tuning a model may be attractive, but needs experts in deep learning tuning. We are currently more seeing a library of prompts and a small set of reliable LLMs that developers can trust. 

There will be still developers, data scientists, product managers, to develop solution around LLMs. A solution will use different LLMs and different capabilities to support multiple requirements that a business application needs: summarization, Q&A, chatbot, translation for example will mostly be combined in a lot of enterprise solutions.

Data enrichment, prompt engineering, user interface, deployment, HA, multi tenancy, security, decision governance may all be part of any AI solution.

[Deeplearning.ai](https://www.deeplearning.ai/) proposes the following LLM project life cycle:

![](./diagrams/llm-project.drawio.png)

### Gen AI Specific Scoping

1. Go over the [discovery assessment](#discovery-assessment)
1. Define what the key metrics are and how to evaluate the solution. If the use cases fall into the Document Q&A and Document Summarization categories, the metric used will be **accuracy**. **Accuracy** will be determined based on the documents (data) provided and the respective questions users ask against the model.
1. Define a list of questions that we expect the application to answer. Be sure to have a list of correct answers. In case of summarization use cases, we need sample summaries and sample questions to generate those summaries for document summarization use cases.


### Model Evaluation

There are web sites to evaluate existing LLMs, but they are based on public data, and may not perform well in the context of a specific use case with private data.

The methodology looks like in the following steps:

* Select models based on specific use case and tasks
* Human calibration of the models: understand behavior on certain tasks, fine tune prompts and assess against a ground truth using cosine-sim. Rouge scores can be used to compare summarizations, based on statistical word similarity scoring.
* Automated evaluation of models: test scenario with deep data preparation, was is a good answer. LLM can be used as a judge: variables used are accuracy, coherence, factuality, completeness. Model card
* ML Ops integration, self correctness

Considerations

* Licensing / copyright
* Operational
* Flexibility
* Human language support

### Consumers of LMs

This is the category of application that consumes pre-trained models to generate text, image, videos, audio or code.

### Discovery Assessment

When engaging with a customer it is important to assess where they are in their GenAi adoption. Think big about the opportunities, but start small with problems that cause day-to-day irritations for the employees or customer.

???- question "Research for opportunities"
    * What manual or repetitive processes could be automated with generative AI?
    * Where do employees spend the most time today gathering information to do their jobs?
    * What customer/user pain points could be addressed with more natural conversation?
    * What content creation processes could be enhanced through AI generated drafts?
    * What expert skills are scarce in your organization that AI models could supplement?
    * What insights could be uncovered from large volumes of unstructured data using AI?
    * What risks or inefficiencies exist from decisions made with incomplete information?
    * Where does communication break down today between teams, customers or regions?
    * What predictions would help you make smarter real-time decisions?
    * What new products, services or business models could AI capabilities enable?
    * What tasks or processes still rely heavily on tribal knowledge that could be systematized?
    * What information gets trapped in siloed systems and could be unlocked with AI?
    * What customer research efforts could be accelerated with interactive AI agents?
    * What compliance processes result in slowdowns getting products/services to market?

???- question "Use cases and business needs"
    * What are the potential use cases? B2B, B2C, Employees?
    * Is the use case considered a strategic priority? Sponsor?
    * What is the value associated with the use case?
    * Are subject matter experts available to support the use case?
    * Who is the end user?
    * What are the current user's challenges and pains?
    * What will be the "ha-ha" moment for the user?
    * Do you have data sets? Quality?

???- question "Experience in AI"
    * Are you using AI in your current business applications, or processes?
    * What are current/past successes by adopting AI in the business execution?
    * What is the current level of ML support needed for your technical staff?
    * Do you need capabilities like summarization, text generation, speech recognition?
    * How will you monitor model performance and detect model drift over time?


???- question "Generative AI current adoption"
    * How familiar with Generative AI? and its common use cases?
    * What GenAI technologies have you/are you evaluating?
    * Have you started prototyping?
    * Do you have AI-powered products or features on your roadmap?
    * Do you have tried to use an existing generative models? to tune it?
    * What are the current process to evaluate Gen AI models?
    * What is your risk appetite for model hallucination and its potential consequences?
    * How do you plan to do domain  adaptation? Do you plan to pre-train, fine-tune or do some in-context  prompting for domain adaptation?
    * How frequently data changes? How frequently do you expect to need to retrain models with new data?


???- question "Integration needs"
    * Is it a new solution or extending an existing one?
    * Where data coming from?
    * What type of systems to integrate the solution with? 
    * Any expected performance requirements? 

???- question "Security and compliance needs"
    * Code privacy and IP related code control