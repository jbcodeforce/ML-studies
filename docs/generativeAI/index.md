# Generative AI


Ways to use Gen AI

* Build foundation model from scratch
* Reuse existing foundation models available as open-source (Hugging Face) or proprietary.

???- "Hugging Face"
    [Hugging Face](https://huggingface.co/) is an open-source provider of natural language processing (NLP), which makes it easy to add state of the art ML models to  applications. We can deploy and fine-tune pre-trained models reducing the time it takes to set up and use these NLP models from weeks to minutes.

## Use case

* Documentation summarization: Start for example for AI21 Jurassic-2 Jumbo.
* Question/answer solution based on internal documents.
* Model fine tuning to downsize the LLM to limit inference cost.
* Transcription insights, by extracting action items from the videos.
* Chat functionality with context.

## ChatGPT

Chat Generative Pretrained Transformer is a system of models designed to create human like conversations.


## Dolly

[Dolly 2.0](https://huggingface.co/databricks/dolly-v2-12b) is built the first open source, instruction-following Large Language Model by Databricks in partnership with Huggingface. It comes with 3b or 12b parameters trained models. 

The model is based on the [eleuther](https://www.eleuther.ai/) (a non-profit AI research lab focusing on Large models) with 6b parameter model named Pythia. On top of this model, Databricks created human generated prompts (around $15k). 

[ChatGPT](https://openai.com/blog/chatgpt), a proprietary instruction-following model, was released in November 2022. The model was trained on trillions of words from the web, requiring massive numbers of GPUs to develop. The model was trained using Reinforcement Learning from Human Feedback (RLHF), using the same methods as [InstructGPT](), but with different data collection setup. 

