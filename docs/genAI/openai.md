# [OpenAI](https://platform.openai.com/docs/overview)

The offering includes [ChatGPT URL](https://chat.openai.com), models (GPT-4, DALL.E, TTS, Whisper, Embeddings), SDK, APIs.

## Notes

* personal data is not used to train or improve the models
* API data may be retained for up to 30 days. Except customers willing to do zero data retention.

## Quickstart

Once we got an API key, it is important to understand the [API limits](https://platform.openai.com/account/limits) and [pricing](https://openai.com/pricing#language-models).

## Assistants API

[Assistants](https://platform.openai.com/docs/assistants/overview) can leverage models and tools like Code Interpreter (OpenAI hosted), Retrieval (OpenAI hosted), and Function calling, to respond to user queries.

Assistants are created via API by specifying its name, instructions, a set of tools and then the LLM model name to use.

OpenAI uses the concept of Thread to represent a conversation between a user and one or many Assistants. Messages are added to the Thread. The Thread is explicitly executed with a run.

Threads simplify AI application development by storing message history and truncating it when the conversation gets too long for the model’s context length.

```python
from openai import OpenAI
client = OpenAI()
  
assistant = client.beta.assistants.create(
  name="Math Tutor",
  instructions="You are a personal math tutor. Write and run code to answer math questions.",
  tools=[{"type": "code_interpreter"}],
  model="gpt-4-turbo-preview",
)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=".... Can you help me?"
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please ..."
)
```

Code Interpreter may execute our own nodejs or python code on OpenAI hosts.

The Assistants API automatically manages the context window such that you never exceed the model's context length.

Run has states: Queued, In progress, requires actions, expired, completed, failed, cancelling, cancelled.
We can poll the run states. 
