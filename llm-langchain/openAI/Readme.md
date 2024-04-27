# OpenAI examples

All the code access env variables, like OPENAI_API_KEY, with dotenv module, be sure to have the .env in the ML-studies folder. The following code can be run with python 3.11 in the openAI folder.

## Code reference

| Example | Purpose |
| --- | --- |
| 1st_openAI_lc.py | A simple chain to ask about LangSmith and testing. Which should generate hallucination as ChatGPT was cutoff in 2023 |
| agent_memory| use tool and chat_history to keep memory of the conversation |
| agent with tools retriever | tool descriptions are added to a vector store. with vector store retreiver we can get the relevant document / tool to support the query |
| build_agent_domain | load the blog on multi agents and save it in vector store persisted on disk for other apps | 
| client stream | to test the API of work_struct_output_for_tool |
| conversation with memory | use memory, window memory and short term memory to keep more context of a conversation | 
| extract json from text | Use prompt to extract data from a customer review | 
| json_agent | Tool calling for last news Tabily search, using a React Prompt (using question, thought, action, observation) and create_json_chat_agent |
| multi chains | sequential chains and router chain demo |
| openAI_agent.py | An agent using retriever and Tavily Search as tools for tool calling llm |
| openai api | use open ai api directly without langchain |
| openAI_chat_lc.py | Use the Langchain user guide, FAISS and use chat history |
| openAI_retrieval_lc.py | Add Retriever to get better data, by crawling a LangChain product documentation from the web using BeautifulSoup then FAISS vector store. It uses `langchain.chains.combine_documents. create_stuff_documents_chain` |
| query agent domain store | Query the domain store, with a text chat |
| simple_client_OpenAI_api | use openai api directly . no langchain |
| streamin_ouput.py | testing the model streaming api |
| text_summarization.py | Summarization of text from tecton.ai blog |
| web_server_wt_streaming | different examples of streaming with fastapi and chain or agent |
| work_struct_output_for_tool |  tool calling with structured response and custom parser |


## Some specific considerations

