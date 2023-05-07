# Generative AI


Some ways to use Gen AI:

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

## Concepts

| Term | Definition |
| --- | --- |
| Transformer |	A ML model for transforming one sequence into another, using attention.|
| Attention	| A math filter for focusing on the important parts of data inputs. |
| Large Language Model (LLM’s) |	Transformers trained on millions of documents |
| Foundation Model | Original models, trained at large expense. |
| Fine Tuning	| Foundation model further trained to specific tasks. Example: training BLOOM to summarize chat history where you have examples of these text examples. |
| Pretraining	| Unsupervised learning method which is used to steer foundation models to domain specific information. Example: pretraining FLAN with Medical documents to understand medical context previously missing from the model. |
| Transfer learning	| The act of transferring the power of a foundation model to your specific task. |
| AI21 Labs	| AI21 Studio provides API access to Jurassic-2 large language models. Their models power text generation and comprehension features in thousands of live applications. AI21 is building state of the art language models with a focus on understanding meaning. |
| co:here |	Co:here platform can be used to generate or analyze text to do things like write copy, moderate content, classify data and extract information, all at a massive scale. |
| Stability.ai	| Stability AI is open source generative AI company currently developing breakthrough AI models applied to imaging, language, code, audio, video, 3D content, design, biotech. Stability AI's partnership with AWS provides the world’s fifth-largest supercomputer – the Ezra-1 UltraCluster – supplying the necessary power to generate these advancements. Stability AI’s premium imaging application DreamStudio, alongside externally built products like Lensa, Wonder and NightCafe, have amassed over 40 million users and counting. |
| FLAN	| FLAN(Fine-tuned LAnguage Net): is a LLM with Instruction Fine-Tuning. It is a popular open source instructor based model which scientists can train. Persons who want an open source alternative to GPT might look at this. |
| BLOOM	| BLOOM is an autoregressive Large Language Model (LLM), trained to continue text from a prompt on vast amounts of text data using industrial-scale computational resources. As such, it is able to output coherent text in 46 languages and 13 programming languages that is hardly distinguishable from text written by humans. BLOOM can also be instructed to perform text tasks it hasn't been explicitly trained for, by casting them as text generation tasks. It is a popular open source instructor based model. Developers who want an open source alternative to GPT might look at this. |
| GPT	| OpenAI's generalized pretrained transformer foundation model family. GPT 1 and 2 are open source while 3 and 4 are propietary. GPT1,2,3 are text-to-text while gpt4 is multimodal. |
| BERT	| Bidirectional Encoder Representations from Transformers (BERT) is a family of masked-language models published in 2018 by researchers at Google. It is much smaller than current LLMs, so if the task can be accomplished by BERT it can be very helpful for developers - however it usually does not perform as well as other foundaiton models because it is not large enough. |
| Davinci	| OpenAI's GPT3 text-to-text based model. It is proprietary and only available by API. People can fine tune this model on OpenAI.|
| Jurassic	| This is AI21 lab's foundation text to text model. It has instructor and non-instructor based versions and is available on AWS marketplace. This is very appealing for customers because they can get 1) extermely high model quality/accuracy and 2) deploy the model to a dedicated endpoint for dedicated compute.
| HuggingFace |	Hugging face makes it easy to add state of the art ML models to  applications. An open-source provider of natural language processing (NLP) models known as Transformers, reducing the time it takes to set up and use these NLP models from weeks to minutes. |
| Prompt engineering | Prompt involves instructions and context passed to a language model to acheive a desired task. Prompt Engineering is a practice of developing and optimizing prompts to efficiently use Lang models for a varierty of applications. Prompt engineering typically works by converting one or more tasks to a prompt-based dataset and training a language model with what has been called "prompt-based learning" or just "prompt learning". |
| MultiModal Models	| Multimodal learning attempts to model the combination of different modalities of data, often arising in real-world applications. An example of multi-modal data is data that combines text (typically represented as discrete word count vectors) with imaging data consisting of pixel intensities and annotation tags. |
| Instructor Models	 | Instructor models are trained specifically to handle instructions-only prompts ("zero-shot") without examples ("few-shot"). This reduces the requirement of curating multiple high quality text examples for single shot inference. |
| Distributed Training | In distributed training the workload to train a model is split up and shared among multiple mini processors, called worker nodes. These worker nodes work in parallel to speed up model training. |
| Retrieval augmented generation (RAG)| The act of supplementing generative text models with data outside of what it was trained on. For example if a model was trained on data up until 2021 and was asked "who is the prime minister of the UK" it might say "Theresa May", but if the model was asked "who is the prime minister of the UK <context of google search here>" it would be able to answer corrrectly with the most up to date information. This is extendable to businesses who want to include information which was not previously used in a foundation model training set but does have the ability to search. Technical documentation which is not public is a good example of this. |
| Generative question and answering |	The new and improved retrieval augmented generation (RAG) |
| Pinecone | A sparse dense vector database which can be used to store sentence embeddings and then utilize approximate nearest neighbor search to fine similarity matches. This can be used for semantic search (search which matches the meaning) and then applied as 'context' to LLMs for question and answering. |
| OpenAI| OpenAI is an AI research and deployment company. Their vision: intelligence—AI systems are generally smarter than humans: 1)With broad general knowledge and domain expertise, GPT-4 can follow complex instructions in natural language and solve difficult problems with accuracy. 2)DALL·E 2 can create original, realistic images and art from a text description. It can combine concepts, attributes, and styles. 3) Whisper can transcribe speech into text and translate many languages into English. |
| Model Distribution | When a model's size prohibits it from being stored on one GPU, the single model has to be stored on more than one GPU. This occurs when models start to be in the 10's of billions of parameter range. This has a few consequences 1) it costs a lot to train and host these models 2) specialized libraries are required to help with this. | 
| Data Distributed Training	| A distributed training algorithm which can speed up ML training by distributing batches of data between forward and backward passes in a model. This can be very helpful when you have large datasets but does not solve the problem of not being able to fit a model on one machine (see model distribution) |
| Model compilation | Model compilation is the act of tracing a model computational graph in order to deploy to lower level hardware and code. This is a necessary step to run on specialized hardware like inferentia and trainium. It can be very finicky. |
| Reinforcement learning with human feedback (RLHF) | The secret sauce to making chat based foundation models. The process involves using human feedback with LLM chat interactions to inform a reinforcement learning procedure to help train an LLM to "talk to humans" instead of only prompts. I think of this as providing two huge benefits 1) this substantially reduces the amount of prompt engineering required and 2) this allow the LLM to take into account chat context as well as the information it has available to it. |
| BARD | AI chat service from google - powered by the LaMDA model. Similar to ChatGPT. |
| Text to text | Any model which takes in text inputs and produces text outputs. Ex: entity extraction, summarization, question answer. |
| Embeddings | Vector representations of non-vector data including images, text, audio. Embeddings allow to perform mathematical operations on otherwise non-mathematical inputs. For example: what is the average of the previous two sentences? |
| Single shot learning | *Zero-shot learning* (ZSL) is a problem setup in ML where, at test time, a learner observes samples from classes which were not observed during training, and needs to predict the class that they belong to | 
| Few shot Learning | *Few-shot learning* or *few-shot prompting* is a prompting technique that allows a model to process examples before attempting a task. |
| DeepSpeed | DeepSpeed is an open source deep learning optimization library for PyTorch. The library is designed to reduce computing power and memory use and to train large distributed models with better parallelism on existing computer hardware. DeepSpeed is optimized for low latency, high throughput training. It can be used on SageMaker to help both inference and training of large models which don't fit on a singel GPU. |
| LangChain | LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.The core idea of the library is that we can “chain” together different components to create more advanced use cases around LLMs. For example, LangChain assits with retieval augmented generation. A common flow for LangChain would be 1) get input from user 2) search relevant data 3) engineer the prompt based on the data retrieved 4) prompt a zero shot instructor model 5) return the output to the user. This is very popular with AWS customers today. |
| LaMDA	| Language model was trained on dialogue from Google. Very similar to ChatGPT but produced by Google. It is a proprietary model. |
| Stable Diffusion | Stable diffusion is a popular open source text to image generation tool. It can be used for use cases like 1) marketing content geration 2) game design 3) fashion design and more. |
| Llama | A foundational, 65-billion-parameter large language model created by Facebook which has been open sourced for academic use. The weights have been leaked and have been found on torrents around the web.  Note that many models have been released based on this, but they also inherit the licencing requrment for non-comertial use. |
| Generative adversarial network (GAN) |	A deep learning architecture where two networks compete in a zero sum game. When one network wins, the other loses and vice versa. Common applications of this include creating new datasets, image generation, and data augmentation. This is a common design paradimn for generative models. |


## ChatGPT

Chat Generative Pretrained Transformer is a system of models designed to create human like conversations.


## Dolly

[Dolly 2.0](https://huggingface.co/databricks/dolly-v2-12b) is built the first open source, instruction-following Large Language Model by Databricks in partnership with Huggingface. It comes with 3b or 12b parameters trained models. 

The model is based on the [eleuther](https://www.eleuther.ai/) (a non-profit AI research lab focusing on Large models) with 6b parameter model named Pythia. On top of this model, Databricks created human generated prompts (around $15k). 

[ChatGPT](https://openai.com/blog/chatgpt), a proprietary instruction-following model, was released in November 2022. The model was trained on trillions of words from the web, requiring massive numbers of GPUs to develop. The model was trained using Reinforcement Learning from Human Feedback (RLHF), using the same methods as [InstructGPT](), but with different data collection setup. 

