# Generative AI

Create new content (text, image,..) from existing one and a requested query. It is based on Large Language Model, pre-trained on huge amount of documents, using 500B of parameters. 

For generative AI, the input is very ambiguous, but also the output: there is no determinist output.  With classical ML output is well expected. 


Some ways to use Generative AI:

* Build foundation model from scratch
* Reuse existing foundation models available as open-source (Hugging Face) or proprietary, add your corpus on top of it.
* Use generative AI services or APIs offered by foundation model vendors. There is not control over the data, cost and customization.

???- "Hugging Face"
    [Hugging Face](https://huggingface.co/) is an open-source provider of natural language processing (NLP), which makes it easy to add state of the art ML models to  applications. We can deploy and fine-tune pre-trained models reducing the time it takes to set up and use these NLP models from weeks to minutes.

## Use cases

* Documentation summarization: See model like Jurassic-2 Jumbo  from [AI21 studio](https://www.ai21.com/studio).
* Question/answer solution based on internal documents.
* Model fine tuning to downsize the LLM to limit inference cost.
* Transcription insights, by extracting action items from the videos.
* Chat functionality with context, with better user's experiences
* Autogeneration of marketing material, translation, summary...
* Self service tutor based on student progress, prompt activities, and respond  to questions
* Synthetic data generation, to keep the privacy of original data sources, and help trains other models: generate image of rusted pumps to train an anomaly detection model on pumps. 
* [Generative Adversarial Networks](https://towardsai.net/p/l/gans-for-synthetic-data-generation) are used to limit the risk of adversarial manipulation in deep learning image recognition. It attempts to generate fake data that looks real by learning the features from the real data.

### Industries

* Supply chain by improving the visibility to multi-tier supplier performance concerns, understand where risk can be found in the supply chain
* Nonconformance/quality dispositioning, by identifying root cause of nonconformance, and prescribe resolutions
* Engineering cost optimization by reusing common parts across plaforms.
* Automation for proofreading, updating databases, managing ads campaigns, analyzing customer reviews, monitoring social media platforms
* Sentiment analysis
* Content moderation and development for education and universities. Helps students to find the most effective pathways to graduation.
* From identifying potential safety risks with gaz leaks, Gen AI can generate recommendations for remedial work. 
* Enhance trip planning with personalized recommendations, services and offers for travel industry.
* product review summarization: today done by human, can be offloaded by LLM by adding those unstructured reviews as new corpus for search. Separate these reviews based on user-provided ratings and task an LLM to extract different sets of information from each high-level category of reviews.

### Discovery

When engaging with a customer it is important to assess where they are in their GenAi adoption:

* How familiar with Generative AI?
* Experienced adapting an existing generative models?
* What are the potential use cases?
* What is current success by adopting AI in their business execution
* Code privacy and IP related code control

## Concepts

A LLM is part of the evolation of NLP as it is a trained deep learning model that understand and generates text in a human like fashion. Deep learning allows a neural network to learn hierarchies of information in a way that is like the function of the human brain. 
From 2017, many NLP models are based on transformers. Which is a neural-network that take into account an entire sentence or paragraph at once instead of what word at a time. It better understands the context of a word. 

To process a text input with a transformer model, we first need to **tokenize** it into a sequence of words. These tokens are then **encoded** as numbers and converted into **embeddings**, which are vector-space representations of the tokens that preserve their meaning. Next, the encoder in the transformer transforms the embeddings of all the tokens into a **context vector**. Using this vector, the transformer decoder generates output based on clues. The decoder can produce the subsequent word. We can reuse the same decoder, but this time the clue will be the previously produced next-word. This process can be repeated to create an entire paragraph.
This process is called **autoregressive generation**.

Transformers do not need to code the grammar rules, they acquire them implicitly from big corpus.

During the training process, the model learns the statistical relationships between words, phrases, and sentences, allowing it to generate coherent and contextually relevant responses when given a prompt or query.

The techniques to customize LLM applications from simplest to more complex. 

* Zero-shot inference.
* Prompt engineering with zero-shot inference.
* Prompt engineering with few-shot inference.
* Retrieval augmented generation (more complex).
* Fine tune an existing foundation model.
* Pre-train an existing foundation model: example is domain specific model, like the Bloomberg's one. 
* Build a foundation model from scratch. 
* Support human in the loop to create high quality data sets.

???- "Zero-shot inference"
    Zero-shot learning in NLP allows a pre-trained LLM to generate responses to tasks that it hasn’t been specifically trained for. In this technique, the model is provided with an input text and a prompt that describes the expected output from the model in natural language.
    **Few-shot** learning involves training a model to perform new tasks by providing only a few examples. This is useful where limited labeled data is available for training. 

### Prompt engineering

A prompt is an input that the model uses as the basis for generating a text. Prompts are a way to directly access the knowledge encoded in large language models. While all the information may be codes in the model, the knowledge extraction can be a hit or miss.

Prompt involves instructions and context passed to a language model to acheive a desired task. 

**Prompt Engineering** is a practice of developing and optimizing prompts to efficiently use LLMs for a varierty of applications. It is still a major research topic.

Prompt engineering typically works by converting one or more tasks to a prompt-based dataset and training a language model with what has been called "prompt-based learning" or just "prompt learning". 

We can provide a prompt with examples so the LLM will condition on the new context to generate better result. Examples in summarization.

Prompts can also help incorporate domain knowledge on specific tasks and improve interpretability. Creating high-quality prompts requires careful consideration of the task at hand, as well as a deep understanding of the model’s strengths and limitations.

LLMs are very sensitive to small perturbations of the prompt: a single typo or word change can alter the output.

There is still need to evaluate models robustness to prompt. 

Many recent LLMs are fine-tuned with a powerful technique called **instruction tuning**, which helps the model generate responses to prompts without prompt-specific fine-tuning. It does not involve updating model weights.

???- "Instruction tuning"
    Technique to train the model with a set of input and output instructions for each task (instead of specific datasets for each task), allowing the model to generalize to new tasks that it hasn’t been explicitly trained on as long as prompts are provided for the tasks. It helps improve the accuracy and effectiveness of models and is helpful in situations where large datasets aren’t available for specific tasks.

### Retrieval augmented generation (RAG)

The act of supplementing generative text models with data outside of what it was trained on. This is extendable to businesses who want to include proprietary information which was not previously used in a foundation model training set but does have the ability to search. Technical documentation which is not public is a good example of this.

The following diagram illustrates a classical RAG process using AWS SageMaker and OpenSearch.

![](./diagrams/rag.drawio.png)

And a classical RAG with LangChain

![](../coding/diagrams/rag-process.drawio.png)

### Important Terms

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
| MultiModal Models	| Multimodal learning attempts to model the combination of different modalities of data, often arising in real-world applications. An example of multi-modal data is data that combines text (typically represented as discrete word count vectors) with imaging data consisting of pixel intensities and annotation tags. |
| Distributed Training | In distributed training the workload to train a model is split up and shared among multiple mini processors, called worker nodes. These worker nodes work in parallel to speed up model training. |
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
| [LangChain](../coding/langchain.md) | LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.The core idea of the library is that we can “chain” together different components to create more advanced use cases around LLMs. For example, LangChain assits with retieval augmented generation. A common flow for LangChain would be 1) get input from user 2) search relevant data 3) engineer the prompt based on the data retrieved 4) prompt a zero shot instructor model 5) return the output to the user. This is very popular with AWS customers today. |
| LaMDA	| Language model was trained on dialogue from Google. Very similar to ChatGPT but produced by Google. It is a proprietary model. |
| Stable Diffusion | Stable diffusion is a popular open source text to image generation tool. It can be used for use cases like 1) marketing content geration 2) game design 3) fashion design and more. |
| Llama | A foundational, 65-billion-parameter large language model created by Facebook which has been open sourced for academic use. The weights have been leaked and have been found on torrents around the web.  Note that many models have been released based on this, but they also inherit the licencing requrment for non-comertial use. |
| Generative adversarial network (GAN) |	A deep learning architecture where two networks compete in a zero sum game. When one network wins, the other loses and vice versa. Common applications of this include creating new datasets, image generation, and data augmentation. This is a common design paradimn for generative models. |


## Current Technology Landscape

### [ChatGPT](https://openai.com/blog/chatgpt)

Chat Generative Pretrained Transformer is a proprietary instruction-following model, was released in November 2022. It is a system of models designed to create human like conversations and generating text by using statistics. It is a Causal Language Model (CLM) trained to predict the next token.

The model was trained on trillions of words from the web, requiring massive numbers of GPUs to develop. The model was trained using Reinforcement Learning from Human Feedback (RLHF), using the same methods as [InstructGPT](https://en.wikipedia.org/wiki/GPT-3), but with different data collection setup. 

### [Amazon Bedrock](https://aws.amazon.com/bedrock/)

[See techno summary in AWS studies.](https://jbcodeforce.github.io/aws-studies/ai-ml/bedrock/)

### Amazon SageMaker

[SageMaker Jumpstart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html) provides pretrained, open-source models for a wide range of problem types to get started on ML.

It supports training on LLMs not in Bedrock, like [OpenLLama](https://github.com/openlm-research/open_llama), [RedPajama](https://github.com/togethercomputer/RedPajama-Data), [Mosaic Mosaic Pretrained Transformer-7B](https://www.mosaicml.com/blog/mpt-7b), [Flan-T5/UL2](https://huggingface.co/docs/transformers/main/model_doc/flan-ul2), [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b), [NEOX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b) and [Bloom/BloomZ](https://huggingface.co/bigscience/bloom), with a gain of up to 40% faster.


* [Quickly build high-accuracy Generative AI applications on enterprise data using Amazon Kendra, LangChain, and large language models.](https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/)
* [SageMaker my own study](https://jbcodeforce.github.io/aws-studies/ai-ml/sagemaker/).

### [Amazon CodeWhisperer](https://aws.amazon.com/codewhisperer/)

[Special studies.](https://jbcodeforce.github.io/aws-studies/coding/#codewhisperer)

### Databricks Dolly

[Dolly 2.0](https://huggingface.co/databricks/dolly-v2-12b) is built the first open source, instruction-following Large Language Model by Databricks in partnership with Huggingface. It comes with 3b or 12b parameters trained models. 

The model is based on the [eleuther](https://www.eleuther.ai/) (a non-profit AI research lab focusing on Large models) with 6b parameter model named Pythia. On top of this model, Databricks created human generated prompts (around $15k). 

## Model Evaluation 

There are web sites to evaluate existing LLMs, but they are based on public data, and may not perform well in the context of a specific use case with private data.

The methodology looks like:

* Downselect models based on specific use case and tasks
* Human calibration of the models: understand behavior on certain tasks, fine tune prompts and assess against a ground truth using cosine-sim. Rouge scores can be used to compare summarizations, based on statistical word similarity scoring.
* Automated evaluation of models: test scenario with deep data preparation, was is a good answer. LLM can be used as a judge: variables used are accuracy, coherence, factuality, completeness. Model card
* ML Ops integration, self correctness

Considerations

* Licensing / copyright
* Operational
* Flexibility
* Language support

## Some interesting readings

* [Vulnerabilities of LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/).
* [GANs for Synthetic Data Generation.](https://towardsai.net/p/l/gans-for-synthetic-data-generation)
* [Artificial Intelligence and the Future of Teaching and Learning](https://www2.ed.gov/documents/ai-report/ai-report.pdf).
* [Fine-tune a pretrained model HuggingFace tutorial](https://huggingface.co/docs/transformers/training).
* [Prompt engineering is the new feature engineering.](https://www.amazon.science/blog/emnlp-prompt-engineering-is-the-new-feature-engineering)
* [Amazon-sponsored workshop advances deep learning for code.](https://www.amazon.science/blog/amazon-sponsored-workshop-advances-deep-learning-for-code)

@huggingfacecourse