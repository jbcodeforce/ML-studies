# AI discussions

The goal of this section is to get a set of content to support deeper discussions around Gen AI, during chit-chat or interviews.

See also the structured learning path in [Guide for AI](../guide_for_ai.md) (Phases 4–5 cover Gen AI and agentic systems).

## 1. Explain LLM fundamentals

Cover the high-level workings of models like GPT-3, including transformers, pre-training, fine-tuning, etc.

* [x] [General LLM introduction](./index.md/#introduction)
* [x] [Transformer and GPT-3 summary](./index.md/#transformer-architecture)
* [x] [How LLM pre-training is done](./index.md/#pre-training-process)
* [x] [How to fine tune existing model](./index.md/#model-fine-tuning)
* [x] [How RAG works](./rag.md)
* [x] [Embeddings and NLP processing](../ml/nlp.md#embedding)
* [x] [Inference parameters (temperature, top-p)](./index.md/#common-llm-inference-parameters)

???- code "Code samples (src/)"
    | Example | Path | Purpose |
    | --- | --- | --- |
    | OpenAI API client | [openai_api.py](../../src/llm-langchain/openai/openai_api.py) | Direct OpenAI API usage without LangChain. See [OpenAI readme](../../src/llm-langchain/openai/Readme.md) |
    | First LangChain chain | [first_openai_lc.py](../../src/llm-langchain/openai/first_openai_lc.py) | Basic chain; illustrates knowledge cut-off / hallucination |
    | OpenAI retrieval | [openai_retrieval_lc.py](../../src/llm-langchain/openai/openai_retrieval_lc.py) | Crawl docs, FAISS vector store, retriever chain |
    | HuggingFace embeddings | [embeddings_hf.py](../../src/llm-langchain/rag/embeddings_hf.py) | Embedding model usage |
    | Bedrock access | [test_bedrock_access.py](../../src/llm-langchain/bedrock/test_bedrock_access.py) | AWS Bedrock client setup |
    | Ollama local LLM | [llama_lc.py](../../src/llm-langchain/ollama/llama_lc.py) | Local model via Ollama. See [Ollama readme](../../src/llm-langchain/ollama/README.md) |

## 2. Discuss prompt engineering

Talk through techniques like demonstrations, examples, and plain language prompts to optimize model performance.

* [Prompt Engineering](./prompt-eng.md)
* [Zero-shot and few-shot prompting](./prompt-eng.md/#prompting-techniques)
* [Chain of Thought (CoT)](./prompt-eng.md/#chain-of-thought)
* [Prompt chaining and Tree of Thoughts](./prompt-eng.md/#prompt-chaining)
* [Automatic Prompt Engineering (APE)](./prompt-eng.md/#automatic-prompt-engineering)
* [ReAct prompting](../coding/langgraph.md#reasoning-and-acting-react-implementation)

???- code "Prompt engineering code (src/ and e2e-demos/)"
    | Example | Path | Purpose |
    | --- | --- | --- |
    | CoT with Bedrock | [test_bedrock_cot.py](../../src/llm-langchain/bedrock/test_bedrock_cot.py) | Chain-of-thought prompts; sample inputs in [cot3.txt](../../src/llm-langchain/bedrock/cot3.txt) |
    | Program-Aided Language | [test_pal_with_claude.py](../../src/llm-langchain/bedrock/test_pal_with_claude.py) | PAL prompting with Claude on Bedrock |
    | Prompt builder graph | [prompt_builder_graph.py](../../src/llm-langchain/langgraph/prompt_builder_graph.py) | LangGraph prompt construction |
    | Customer response | [response_to_unhappy_customer.py](../../src/llm-langchain/text_generation/response_to_unhappy_customer.py) | Domain-specific prompt for text generation |
    | Model evaluation | [evaluation.py](../../src/llm-langchain/openai/evaluation.py) | Assess and compare prompt / model outputs |
    | Critical thinking prompt | [think_deeply/](../../e2e-demos/think_deeply/) | 5-stage thinking framework demo. See [e2e readme](../../e2e-demos/README.md#think-deeply) |

## 3. Share LLM project examples

Walk through hands-on experiences leveraging models like GPT-3, Langchain, or Vector Databases.

* [Review RAG positioning, architecture](./rag.md)
* [Advanced RAG techniques](./rag.md) (multi-query, fusion, HyDE)
* [LangChain overview](../coding/langchain.md)
* [LangGraph agent patterns](../coding/langgraph.md)

???- code "RAG and Q&A projects"
    | Example | Path | Purpose |
    | --- | --- | --- |
    | Streamlit RAG demo | [qa_retrieval/Main.py](../../e2e-demos/qa_retrieval/Main.py) | RAG impact on response quality using Lilian Weng agent blog |
    | Build vector store | [build_agent_domain_rag.py](../../src/llm-langchain/rag/build_agent_domain_rag.py) | Index Lilian Weng multi-agent blog into ChromaDB |
    | Query domain store | [query_agent_domain_store.py](../../src/llm-langchain/rag/query_agent_domain_store.py) | Chat against persisted vector store |
    | Multiple queries RAG | [multiple_queries_rag.py](../../src/llm-langchain/rag/multiple_queries_rag.py) | Multi-query retrieval expansion |
    | RAG fusion | [rag_fusion.py](../../src/llm-langchain/rag/rag_fusion.py) | Reciprocal rank fusion across queries |
    | RAG HyDE | [rag_hyde.py](../../src/llm-langchain/rag/rag_hyde.py) | Hypothetical document embeddings |
    | Adaptive RAG | [adaptive_rag.py](../../src/llm-langchain/langgraph/adaptive_rag.py) | LangGraph adaptive retrieval routing |
    | QA pipeline | [qa-pipeline.py](../../src/llm-langchain/qa/qa-pipeline.py) | Bedrock + ChromaDB Q&A chain |
    | Chat with PDF | [chat_with_pdf/app.py](../../e2e-demos/chat_with_pdf/app.py) | End-to-end PDF Q&A app |
    | RAG readme | [rag/README.md](../../src/llm-langchain/rag/README.md) | Index of all RAG examples |

???- code "Agentic project examples"
    | Example | Path | Purpose |
    | --- | --- | --- |
    | Agno overview | [agno.md](./agno.md) | Agno SDK patterns and documentation |
    | Agno studies | [src/agentic/agno/](../../src/agentic/agno/) | Agents with Ollama, tools, knowledge, workflows |
    | Deep researcher | [deep_researcher/](../../src/agentic/agno/deep_researcher/) | Multi-agent research workflow. See [README](../../src/agentic/agno/deep_researcher/README.md) |
    | LLM wiki agent | [llm-wiki/](../../src/agentic/agno/llm-wiki/) | RAG over a personal wiki with Agno |
    | ReAct LangGraph | [react_lg.py](../../src/llm-langchain/langgraph/react_lg.py) | ReAct agent with chain-of-thought prompt |
    | Agent with tools | [openai_agent.py](../../src/llm-langchain/openai/openai_agent.py) | Tool calling with retriever and Tavily search |

## 4. Stay updated on research

Mention latest papers and innovations in few-shot learning, prompt tuning, chain of thought prompting, etc.

* [Few-shot and zero-shot prompting](./prompt-eng.md/#prompting-techniques) — in-context learning without weight updates
* [Instruction tuning](./prompt-eng.md) — fine-tuning on task instructions rather than per-task datasets
* [Chain of Thought](./prompt-eng.md/#chain-of-thought) — intermediate reasoning steps; code in [test_bedrock_cot.py](../../src/llm-langchain/bedrock/test_bedrock_cot.py)
* [Tree of Thoughts](./prompt-eng.md/#tree-of-thoughts) — search over reasoning paths
* [Agentic AI](./agentic.md) — planning, memory, tools, multi-agent patterns
* [Agno framework notes](./agno.md)
* [Model Context Protocol (MCP)](./mcp.md) — standardized tool integration for assistants
* [Hermes agent notes](./hermes.md)

## 5. Dive into model architectures

Compare transformer networks like GPT-3 vs Codex. Explain self-attention, encodings, model depth, etc.

* [Transformer architecture](./index.md/#transformer-architecture) — attention, embeddings, positional encoding, decoder-only vs encoder-decoder
* [Encoder-decoder and generative models](./index.md/#introduction) — three transformer types
* [Transfer learning and fine-tuning](../ml/deep-learning.md#transfer-learning) — reusing pre-trained weights (CV and NLP)
* [Deep learning foundations](../ml/deep-learning.md) — CNNs, training loops, pre-trained model usage
* [PyTorch computer vision](../../src/pytorch/computer-vision/) — CNN training with [fashion_cnn.py](../../src/pytorch/computer-vision/fashion_cnn.py)
* [Distributed training (DDP)](../coding/ddp.md) — multi-GPU training; references minGPT fine-tuning

## 6. Work with Skills

Skills package reusable agent capabilities (prompts, tools, workflows) for AI coding assistants and orchestration platforms.

* At startup, an agent only reads a tiny piece of metadata (the skill's description). It doesn't load the heavy instructions or assets until it explicitly decides the skill is relevant to the user's task.
* A standard skill is packaged as a self-contained folder:
    ```sh
    my-specialized-skill/
    ├── SKILL.md          # Core specification, triggers, & instructions
    ├── scripts/          # Deterministic executable scripts (Python, Bash, etc.)
    └── templates/        # Boilerplate files, assets, or reference docs
    ```

* The `SKILL.md` file includes:
    * **YAML Frontmatter (The Metadata):** Located at the very top of the file. You must provide a crisp `name` and a hyper-focused `description`. Treat this description like regex for the agent's brain—it acts as the trigger condition (Zhang, n.d.).
    * **The Procedural Body:** Written in standard Markdown. This is where you lay out the multi-step, phased workflows, conditional logic, and specific tool execution expectations for the agent (Chen, n.d.).
    * **Specification Tip:** If your description is too broad, the agent will trigger it mistakenly; if it's too narrow, the agent won't reuse it when a slightly different task arises. Striking a balance is key.
* Implement Execution Best Practices:
    * If a step in your workflow requires zero improvisation (e.g., parsing a specific CSV format or calling an internal API), do not write natural language instructions for it. Put it in a script inside the `/scripts` directory and instruct the agent to execute it.
    * **Prevent Plan Drift:** Under small variations, language models inherently want to change up step orders or alter tool arguments. Use clear, sequential step boundaries in your `SKILL.md` to force the agent into a predictable execution loop: *Gather context -> Take action  -> Verify results*.
    * As agents advance, they often branch into parallel sub-agents. Ensure your skill scripts do not cause state collisions if invoked simultaneously.
* Validation & Continuous Evaluation: 
    * Use basic linting to ensure your YAML frontmatter fields are complete and structure layouts comply with standard skill formats.
    * Transition to running paired simulation trials where you evaluate agent trajectories *with* the skill versus a baseline *without* the skill. This is how you objectively measure skill checking if it genuinely improves efficiency, accuracy, and safety.

* [Claude / Cursor agent skills](./agentic.md) — SKILL.md format, `.cursor/skills/` setup
* [Claude Code plugins and skills](../techno/claude/index.md)

* ["Skills Are the New Apps– Now It’s Time for Skill OS" - Le Chen and co](https://www.preprints.org/manuscript/202602.1096)
* [Agentic Continuous Evaluation of Skills (ACES) kevin C](https://openreview.net/forum?id=cf92xtZK47)
* [SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization](https://arxiv.org/abs/2606.06079)

## 7. Discuss fine-tuning techniques

Explain supervised fine-tuning, parameter efficient fine tuning, few-shot learning, and other methods to specialize pre-trained models for specific tasks.

* [Model fine-tuning overview](./index.md/#model-fine-tuning) — when to fine-tune vs RAG vs prompt engineering
* [Transfer learning](../ml/deep-learning.md#transfer-learning) — load pre-trained weights, adapt to custom data
* [NLP embeddings and fine-tuning](../ml/nlp.md) — document-level embedding training
* [Few-shot prompting](./prompt-eng.md/#prompting-techniques) — in-context examples without weight updates
* [Instruction tuning](./prompt-eng.md) — alignment via instruction datasets (RLHF)
* [RAG vs fine-tuning tradeoffs](./rag.md) — cost, quality, and skill requirements
* [Resume tailoring demo](../../e2e-demos/resume_tuning/) — prompt-based specialization to a job description (not weight fine-tuning, but illustrates task adaptation)

## 8. Demonstrate production engineering expertise

From tokenization to embeddings to deployment, showcase your ability to operationalize models at scale, and monitoring model inference.

* [OpenAI streaming and deployment](./openai.md) — API patterns for production
* [Feature stores](../techno/feature_store.md) — Feast and FeatureForm for ML serving
* [Methodology for AI projects](../methodology/index.md) — scoping, risk, and team skills

???- code "Production-oriented examples"
    | Example | Path | Purpose |
    | --- | --- | --- |
    | FastAPI streaming server | [web_server_wt_streaming.py](../../src/llm-langchain/openai/web_server_wt_streaming.py) | Streaming chain/agent behind HTTP |
    | Async streaming | [async_stream.py](../../src/llm-langchain/openai/async_stream.py) | Async OpenAI streaming client |
    | Model evaluation | [evaluation.py](../../src/llm-langchain/openai/evaluation.py) | Output quality assessment |
    | AgentOS | [first_agent_os.py](../../src/agentic/agno/first_agent_os.py) | Agno production agent server |
    | LiteLLM proxy + Prometheus | [claude_code_liteLLM/](../../e2e-demos/claude_code_liteLLM/) | LLM gateway with monitoring. See [README](../../e2e-demos/claude_code_liteLLM/README.md) |
    | Streaming demo | [streaming-demo/](../../e2e-demos/streaming-demo/) | LangGraph streaming UI |
    | Feast feature store | [feast/](../../src/llm-langchain/feast/) | Feature repo and serving examples |
    | FeatureForm | [featureform/](../../src/llm-langchain/featureform/) | Feature definitions and training pipeline |
    | DDP multi-GPU | [multi_gpu_ddp.py](../../src/pytorch/ddp/multi_gpu_ddp.py) | Distributed PyTorch training |
