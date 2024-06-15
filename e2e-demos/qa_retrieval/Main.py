"""
streamlit app to demonstrate RAG using web crawler
"""
import streamlit as st
import bs4
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
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


"""
A basic webapp to demonstrate the impact of RAG on response quality.
Streamlit reruns the whole script each time a widget is updated, so this implementation uses session state
to keep data between tabs
"""
ANTHROPIC="Anthropic-Claude"
OPENAI="OpenAI gpt3.5"
MISTRAL="Mistral"
URL_ARTICLE="https://lilianweng.github.io/posts/2023-06-23-agent/"

load_dotenv("../../.env")

model_name = OPENAI
model_to_use= ChatOpenAI(temperature=.3) 
use_rag = False
DEFAULT_PROMPT="You are a helpful assistant, expert in Information Technology architecture and Generative AI solutions."
current_prompt = DEFAULT_PROMPT
VS_PATH="./chroma_db"
vectorstore =  Chroma(persist_directory=VS_PATH)


# ------------------- RAG related functions --------------------


def process_text(text, vectorstore ):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectorstore.from_texts(texts=chunks, embeddings=embeddings)


def build_indexing(url, vectorstore):
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
    vectorstore.from_documents(documents=splits, embedding=OpenAIEmbeddings())


def define_model(name: str):
    global model_to_use
    if name is ANTHROPIC:
        #model = ChatAnthropic(model='claude-3-opus-20240229')
        model_to_use= ChatAnthropic(model='claude-2.1')
    else:
        model_to_use= ChatOpenAI(temperature=.3) 
    return model_to_use

def define_basic_chain(model_to_use, prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | model_to_use | output_parser
    return chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def change_prompt(use_rag):
    global current_prompt
    
    if not use_rag:
        current_prompt = DEFAULT_PROMPT
    else:
        current_prompt = hub.pull("rlm/rag-prompt")
    #print(f"--> change rag {current_prompt}")
    return current_prompt

def define_rag_chain(model_to_use, prompt, vectorstore):
    retriever = vectorstore.as_retriever()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model_to_use
        | StrOutputParser()
    )

def submit_query(query: str, 
                 mname: str,
                 prompt: str, 
                 useRag: bool = False,
                 vector_store = None):
    global model_to_use

    model_to_use= define_model(mname)
    if useRag:
        chain = define_rag_chain(model_to_use,prompt,vector_store)
    else:
        chain = define_basic_chain(model_to_use, prompt)
    return chain.invoke({"input": query})


# --------------------- UI Code --------------------

def load_prompt():
    with st.spinner("Load new prompt...."):
        st.session_state.use_rag = not use_rag
        prompt=change_prompt( st.session_state.use_rag)
        st.session_state.current_prompt=prompt


def user_interface():
    global vectorstore
    Image.open("images/favicon.ico")

    if "model_name" not in st.session_state:
        st.session_state.model_name = model_name

    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt= current_prompt

    if "user_rag" not in st.session_state:
        st.session_state.use_rag = use_rag


    st.write("# Welcome to RAG Demonstration")
    tab1, tab2, tab3 = st.tabs(["Home", "Context", "Demonstration"])

    with tab1:
        st.markdown(
            """
        This demonstration illustrates the impact of Retrieval Augmented Generation to reduce LLM hallucinations.
    """)
        st.markdown("### Current prompt")
        st.markdown(f"**{st.session_state.current_prompt}**")
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
    1. **Indexing** a batch processing to ingest documents from a source and indexing them. Indexing supports loading the documents, splitting large documents into smaller chunks. Chunks help to stay within the LLM's context window. Indexing includes storage of the documents and index of the splits.
    1. **Retrieval**: retrieves the relevant data (splits) from the index, then passes them to the model as part of the context.
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
    While the chatbot uses the embeddings to get the results of the semantic search close to the current query,

    """)



            

    with tab3:
        st.markdown("# Demonstration")
        st.markdown("## Settings")
        url=URL_ARTICLE
        selected_model = st.selectbox("Possible LLM", key="model_name", options=[ANTHROPIC, OPENAI, MISTRAL])

        st.toggle("Activate Vector Store", key="use_rag", on_change=load_prompt(), args=[use_rag])

        st.markdown("### Current Prompt")
        st.write(st.session_state.current_prompt)


        url=st.text_input("URL for data source",URL_ARTICLE)
        if url is URL_ARTICLE:
            st.image("./images/article.PNG")

        processDoc=st.button("Process this blog to vector store")
        if processDoc:
            with st.spinner("Load doc, and build indexing..."):  
                build_indexing(url, vectorstore)
            st.write("Done !")

        st.markdown("## Query the model:")
        st.write(f" The Model is {st.session_state.model_name}")
        #st.write(f"Using Rag with this prompt:\n{st.session_state.current_prompt}")
        st.write(f"Using Rag:\n{st.session_state.use_rag}")
        query = st.text_input('Ask a question to the LLM',"What is Task Decomposition?")
        submit_btn =  st.button('Submit')
        if submit_btn:
            with st.spinner("Processing..."):
                response= submit_query(query=query,
                                    mname= st.session_state.model_name,
                                    prompt=st.session_state.current_prompt, 
                                    useRag=st.session_state.use_rag,
                                    vector_store=vectorstore)
                st.write(response)


user_interface()