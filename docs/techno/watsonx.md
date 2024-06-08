# WatsonX.ai

[WatsonX.ai](https://www.ibm.com/products/watsonx-ai) offers a set of features to use LLM as APIs and a Studio to train, validate, tune and deploy AI models. 

![](./images/watsonx.ai.main.PNG)

## Value Propositions

* Studio environment to cover both traditional ML model with LLM
* Prompt Lab to build new prompt or use existing ones and shareable between data scientists
* Open sources LLMs and IBM Granite models.
* Support guardrail for model outcome control
* Fine tuning model on proprietary data
* Integrate AutoAI to create ML model in no-code environment
* Ability to create synthetic tabular data
* Open Data lakehouse architecture
* AI governance toolkit
* [Bring your own model](https://www.ibm.com/blog/announcement/bringing-your-own-custom-foundation-model-to-watsonx-ai/)


???- Info "Granite from IBM Research"
    Granite is IBM's flagship series of LLM foundation models based on decoder-only transformer architecture. Granite language models are trained on trusted enterprise data spanning internet, academic, code, legal and finance. [See model documentation.](https://www.ibm.com/products/watsonx-ai/foundation-models#generative)

## Getting started

Once IBM Cloud account is created, we need to also sign-up to WatsonX.ai.

Once done a sandbox project is created, we need to get the project ID using the Info menu in the project page.

![](./images/watsonx-project.png)

* Create an IBM API KEY with Manage (in top menu bar) > Access (IAM) > API keys.
* Get watson.ai the endpoint URL to connect with. Set as environment variables to start using Watson with [LangChain chain](https://python.langchain.com/docs/integrations/llms/ibm_watsonx).
* Get the list of current model for inference:

    ```python
    from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

    print("--> existing models in WatsonX.ai:")
    print(json.dumps( ModelTypes._member_names_, indent=2 ) )
    ```

* Python code to connect to WatsonX ai model

## Prompt Lab

* A shot represent prompt input and output, used to instruct the model on how to best respond to a query
* Prompts are tokenized before being passed into a model, and foundation model usage costs are calculated based on the number of tokens
* WatsonX.ai offers 3 sandbox in the Prompt Lab: chat, structured, freeform

![](./images/prompt-lab-struct.PNG)

* Watsonx.ai provides AI guardrails to prevent potential harmful input and output text
* Watsonx.ai provides sample prompts grouped into categories like: Summarization, Classification, Generation, 
Extraction, Question Answering, Code, Translation.
* It selects the model that is most likely to provide the best performance for the given use case.
* All models have the same inference parameters:

![](./images/model-parameters.PNG)

* In Greedy mode, the model selects the highest probability tokens at every step of decoding. It is less creative. With Sampling we can tune temperature (float), top k(int) and top P (float). Top P sampling chooses from the smallest possible set of “next” words whose cumulative probability exceeds the probability p. The higher the value of Top P, the larger the candidate list of words and so the more random the outcome would be. Top K is for the number of words to choose from to be the output.
* Repetition penalty (1 or 2) is used to counteract a model’s tendency to repeat the prompt text verbatim.

* In general, the **"instruct"** models are better at handling requests for structured output and following instructions.
* Model size does not guaranty better results. IBM's Granite models give excellent results on instruction, like generating, list, json output...

Here is an example of one-shot prompting with an input, output example pair to better guide the model.

![](./images/one-shot-prompt.PNG)

* For better prompt engineering, WatsonX,ai offers save by session to keep a  history of the prompt 
session, recording each individual change, which can also being seen in a timeline. Saving a prompt is like taking a snapshot of the prompt text and its settings. It also support to go back to a previous version.
* A Prompt can be saved as a Jupyter notebook, and then code is generated to run into the notebook.

![](./images/prompt-to-notebook.PNG)

* The **codellama-34b-instruct-hf** is good with code translation and code generation tasks

Foundation models are not answering questions. Instead, they are calculating the best next tokens based on what data was used to train it.


## Prompt tuning

This is not the same as prompt engineering, the goal is to have a user providing a set of labeled data to tune the model, Watsonx.ai will tune the model using this data and create a "soft prompt", without changing the model's weights.

* LLMs are generally not good enough where there are specific business languages and operational details, especially where terminologies and 
business requirements are constantly being updated. 