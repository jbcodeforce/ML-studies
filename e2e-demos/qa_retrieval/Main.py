"""
streamlit app to demonstrate RAG
"""
import os
import streamlit as st
from st_pages import Page, show_pages
import bs4
from pypdf import PdfReader
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from dotenv import load_dotenv
from PIL import Image

from langchain_anthropic import ChatAnthropic

ANTHROPIC="Anthropic-Claude"
OPENAI="OpenAI gpt3.5"
MIXTRAL="Mixtral"
URL_ARTICLE="https://lilianweng.github.io/posts/2023-06-23-agent/"

load_dotenv("../../.env")

model_to_use = None
callWithVectorStore = False
DEFAULT_PROMPT="You are an assistant to get technical support help in IT architecture."
current_prompt = DEFAULT_PROMPT
vectorstore = None

# ------------------- RAG related functions --------------------
def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return Chroma.from_texts(chunks, embeddings)

def indexingDoc(url):
    global vectorstore
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return "Done"

def define_model(name: str):
    global model_to_use
    if name is ANTHROPIC:
        #model = ChatAnthropic(model='claude-3-opus-20240229')
        model_to_use= ChatAnthropic(model='claude-2.1')
    else:
        model_to_use= ChatOpenAI(temperature=.3) 

def define_basic_chain(model_to_use):
    prompt = ChatPromptTemplate.from_messages([
        ("system", current_prompt),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    return prompt | model_to_use | output_parser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def change_prompt(use_rag):
    global current_prompt
    
    if not use_rag:
        current_prompt = DEFAULT_PROMPT
    else:
        current_prompt = hub.pull("rlm/rag-prompt")
    print(f"--> change rag {current_prompt}")

def define_rag_chain(model_to_use):
    global vectorstore
    retriever = vectorstore.as_retriever()
    prompt = current_prompt
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model_to_use
        | StrOutputParser()
    )

def submit_query(query: str, useRag: bool = False):
    global model_to_use
    if model_to_use is None:
        define_model(OPENAI)
    if useRag:
        chain = define_rag_chain(model_to_use)
    else:
        chain = define_basic_chain(model_to_use)
    return chain.invoke({"input": query})


# --------------------- UI Code --------------------
im = Image.open("images/favicon.ico")
st.set_page_config(
    page_title="Introduction",
    page_icon=im,
)

st.write("# Welcome to RAG Demonstration")
tab1, tab2, tab3 = st.tabs(["Home", "Context", "Demonstration"])

with tab1:
    st.markdown(
        """
    This demonstration illustrates the impact of Retrieval Augmented Generation to reduce LLM hallucinations.

**👈 Select the configuration from the left sidebar**!
""")

    st.markdown(f"**Current prompt** is: {current_prompt}")
    st.markdown("""
### Want to learn more?

- See my [ML-Studies website](https://jbcodeforce.github.io/ML-studies/).
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/).
- [Code of this application.](https://github.com/jbcodeforce/ML-studies/blob/master/e2e-demos/qa_retrieval/Main.py)      
"""
)
    
with tab2:
    st.write("# Context")
    st.markdown("## Process")
    st.markdown("The Retrieval Augmented Generation may be seen as a three stages process")
    st.image("./images/rag_3_stages.drawio.png")
    st.markdown("""
1. **Indexing** a batch processing to ingest documents and data from a source and indexing them. During processing semantic search is used to retrieve relevant documents from the index. Indexing supports loading the documents, splitting large documents into smaller chunks. Chunks help to stay within the LLM's context window. Indexing includes storage of the documents and index of the splits.
1. **Retrieval**: retrieves the relevant data (splits) from the index, then passes that to the model as part of the context.
1. **Generation**: generate the response in plain natural language.
""")
 
    st.markdown("## Architecture")
    st.markdown("""
RAG is applied to businesses who want to include proprietary information which was not 
previously part of the model training but does have the ability to search
""")
    st.image("./images/rag.drawio.png")

    st.markdown("""The green part of the diagram above illustrates the ingestion process to build a vector DB.
                It is an off-line processing.
""")
    st.markdown("""
While the chatbot user interface uses the embeddings to get the results of the semantic search close to the current query,

""")

with tab3:
    st.markdown("# Demonstration")
    query = st.text_input('Ask a question to the LLM',"What is Task Decomposition?")
    submit_btn =  st.button('Submit')
    if submit_btn:
        st.write(f"Using Rag with this prompt\n{current_prompt}")
        with st.spinner("Processing..."):
            response= submit_query(query=query,useRag=callWithVectorStore)
            st.write(response)

with st.sidebar:
    url=URL_ARTICLE
    selected_model = st.selectbox("Possible LLM", [ANTHROPIC, OPENAI, MIXTRAL])
    if selected_model is not None:
        define_model(selected_model)

    callWithVectorStore = st.toggle("Activate Vector Store")
    if callWithVectorStore:
        change_prompt(callWithVectorStore)

    url=st.text_input("URL for data source",URL_ARTICLE)
    if url is URL_ARTICLE:
        st.image("./images/article.PNG")

    processDoc=st.button("Process this blog to vector store")
    if processDoc:
        with st.spinner("Processing..."):
            done=indexingDoc(url)
            st.write(done)

    pdf = st.file_uploader('Or upload your Document', type='pdf')
    if pdf is not None:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        kb = process_text(text)



