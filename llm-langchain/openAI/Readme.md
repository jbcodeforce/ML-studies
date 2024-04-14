# OpenAI examples

To get access to the OPENAI_KEY, be sure to have the .env in the ML-studies folder.

| Example | Purpose |
| --- | --- |
| 1st_openAI_lc.py | A simple chain to ask about LangSmith and testing. Which should generate hallucination as ChatGPT was cutoff in 2023 |
| openAI_retrieval_lc.py | Add Retriever to get better data, by crawling a LangChain product documentation from the web using BeautifulSoup then FAISS vector store. It uses `langchain.chains.combine_documents. create_stuff_documents_chain` |
| text_summarization.py | Summarize a blog from Texton.ai | 
| openAI_chat_lc.py | Use the Langchain user guide, FAISS and use chat history |
