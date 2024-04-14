# Prompt engineering

???- info "Upadte"
    Created Aug 2023 - Updated 02/2024

This chapter includes a summary of prompt engineering practices and links to major information on this subject. Main sources of knowledge are:

* [Prompt engineering guide from (promptingguide.ai)](https://www.promptingguide.ai) which covers the theory and practical aspects of prompt engineering and how to leverage the best prompting techniques to interact and build with LLMs.
* [Wikipedia- prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering)
* [Anthropic - Claude - Prompt engineering.](https://docs.anthropic.com/claude/docs/prompt-engineering)

This repository includes some code, prompts to test on different LLM.

## Introduction

A prompt is an input that the model uses as the basis for generating a text. Prompts are a way to directly access the knowledge encoded in large language models. While all the information may be codes in the model, the knowledge extraction can be a hit or miss.

Prompt involves instructions and context passed to a language model to achieve a desired task.

**Prompt Engineering** is a practice of developing and optimizing prompts to efficiently use LLMs for a variety of applications. It is still a major research topic.

Prompt engineering typically works by converting one or more tasks to a prompt-based dataset and training a language model with what has been called "prompt-based learning" or just "prompt learning".

We can provide a prompt with examples so the LLM will condition on the new context to generate better results. Examples in summarization.

Prompts may also help incorporating domain knowledge on specific tasks and improve interpretability. Creating high-quality prompts requires careful consideration of the task at hand, as well as a deep understanding of the model’s strengths and limitations.

LLMs are very sensitive to small perturbations of the prompt: a single typo or word change can alter the output.

There is still need to evaluate models robustness to prompt.

Many recent LLMs are fine-tuned with a powerful technique called **instruction tuning**, which helps the models generate responses to prompts without prompt-specific fine-tuning. It does not involve updating model weights.

???- "Instruction tuning"
    Technique to train the model with a set of input and output instructions for each task (instead of specific datasets for each task), allowing the model to generalize to new tasks that it hasn’t been explicitly trained on as long as prompts are provided for the tasks. It helps improve the accuracy and effectiveness of models and is helpful in situations where large datasets aren’t available for specific tasks.

## Tuning parameters

The classical [Temperature, Top-P](./index.md/#common-llm-inference-parameter-definitions) and max length parameters need to be tuned to get more relevant responses. The Stop sequence, frequency penalty (penalty on next token already present in the response), presence penalty (to limit repeating phrases) are also used in the prompt.

## Prompting techniques

* **Zero-shot prompting:** a unique question. No instruction
* **Few-shot prompting** includes some samples like a list of Q&A. 
* A prompt contains any of the following elements: instruction, context, input data, output indicator
* Use command to instruct the model to do something specific: "Write", "Classify", "Summarize", "Translate", "Order"
* Be very specific about the instruction and task. 
* Providing examples in the prompt is very effective to get desired output in specific formats.

### Chain of Thought

Chain-of-thought ([CoT](https://www.promptingguide.ai/techniques/cot)) prompting using intermediate step.  It is used to address more complex arithmetic, commonsense, and symbolic reasoning tasks. The zero-shot CoT seems to get good results by adding "Let's think step by step" sentence. (see the test code [under llm-langchain/bedrock folder](https://github.com/jbcodeforce/ML-studies/blob/master/llm-langchain/bedrock/TestBedrockCoT.py): `python TestBedrockCoT.py -p cot3.txt -m  ai21.j2-mid`)

* Examples:

```
explain Quantum mechanics to high school student
A:
```

See also the article from Anthropic: ["ask Claude to think step by step"](https://docs.anthropic.com/claude/docs/ask-claude-to-think-step-by-step)

* **RLHF** (reinforcement learning from human feedback) has been adopted to scale instruction tuning wherein the model is aligned to better fit human preferences.

* **Self consistency** prompt uses sample multiple, diverse reasoning paths through few-shot CoT, and use the generations to select the most consistent answer.

* To test CoT with Bedrock LLMs. See the code in [TestBedrockCot](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/bedrock/) with `python TestBedrockCoT.py -p cot1.txt -m anthropic.claude-v2`  should gives good answer. While `python TestBedrockCoT.py -p cot2.txt -m  ai21.j2-mid` returns bad answers.

### [Prompt chaining](https://www.promptingguide.ai/techniques/prompt_chaining)

Break prompt task into subtasks, and add the response of a subtask to the next subtask call to LLM. It creates a chain of prompts. Prompt chaining is useful when building conversational assistants or for document QA, and when the prompt is detailed.

It brings better performance, and helps to boost transparency of the LLM application, increasing controllability, and reliability.

For document QA, a first prompt is used to extract the important quotes from the document given the question, the second prompt uses the generated quotes as input.

See [Anthropic Claude - Prompt Chaining examples.](https://docs.anthropic.com/claude/docs/prompt-chaining)

It can be used to validate a previous response to a prompt.

### [Tree of Thoughts](https://www.promptingguide.ai/techniques/tot) 

Tree of Thoughts is a generalization of CoT, where thoughts represent coherent language sequences that serve as intermediate steps toward solving a problem.
The thoughts are organized in tree, which search algorithms (breadth-first search and depth-first search) are used to assess the best combination of thoughts via a multi-round conversation.

### [Automatic Prompt Engineering](https://www.promptingguide.ai/techniques/ape)

Another approach to automate the prompt creation and selection using LLMs as inference models followed by LLMs as scoring models. [DSPy ](https://github.com/stanfordnlp/dspy) is a framework for algorithmically optimizing LM prompts and weights. 

### Other techniques

* **Directional Stimulus Prompting**: "Summarize the above article briefly in 2-3 sentences based on the hint. Hint:...."

* **Program-Aided Language Models**: use LLMs to read natural language problems and generate programs as the intermediate reasoning steps. See example of such prompt in this python code [TestPALwithClaude.py](ttps://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/bedrock/TestPALwithClaude.py)

* **ReAct Prompting** uses LLMs to generate both reasoning traces and task-specific actions.

## OpenAI

[The best practices for prompt engineering using OpenAI models](https://platform.openai.com/docs/guides/prompt-engineering) can be summarized as:

* Write clear instruction with "ask for brief replies", or "expert-level writing", or demonstrate expected result.
* Provide reference text
* Split complex tasks into simpler subtasks
* Give the model time to "think"


## Playground

* Use Bedrock interface for [text playground](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/text-playground) one of the integrated model.
* [A repository with Bedrock content](https://github.com/aws-samples/amazon-bedrock-workshop.git) with [text generation](https://github.com/aws-samples/amazon-bedrock-workshop/tree/main/01_Generation) notebooks.
* In this repository the [llm-langchain/bedrock](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/bedrock) includes  Python bedrock client codes with prompt samples.

???- Info "How to use Bedrock client"
    The basic instruction to use the code of this repository to interact with an LLM deployed in Bedrock

    * Use virtual Python env

    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    * Install dependencies: `pip3 install -r requirements.txt`
    * Be sure to have  AWS_SESSION_TOKEN set up
    * Run any of the python code. `python TestClaudeOnBedrock.py`

* A [notebook with Mixtral LLM](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/notebooks/pe-mixtral-introduction.ipynb)

