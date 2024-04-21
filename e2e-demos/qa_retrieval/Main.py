"""
streamlit app to demonstrate RAG
"""
import os
import streamlit as st
from st_pages import Page, show_pages

from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import CharacterTextSplitter

from dotenv import load_dotenv
from PIL import Image

from langchain_anthropic import ChatAnthropic


load_dotenv("../../.env")

model_to_use = "Claude"
callWithVectorStore = False
current_prompt="You are an assistant to get technical support help in IT architecture."

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

def submit_query(query: str):
    global model_to_use
    if model_to_use is "Claude":
        #model = ChatAnthropic(model='claude-3-opus-20240229')
        model = ChatAnthropic(model='claude-2.1')
    else:
        model = ChatOpenAI(temperature=.3) 

    prompt = ChatPromptTemplate.from_messages([
        ("system", ),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"input": query})

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

### Want to learn more?

- See my [ML-Studies website](https://jbcodeforce.github.io/ML-studies/)
- [LangChain RAG]()
        
"""
)
    
with tab2:
    st.write("# Context")
    st.markdown("## Architecture")
    st.markdown("""
RAG is applied to businesses who want to include proprietary information which was not 
previously part of the model training but does have the ability to search
""")
    st.image("./images/rag.drawio.png")

    st.markdown("""The green part of the diagram above illustrate the ingestion process to build a vector DB""")
    st.markdown("""
While the chatbot user interface uses the embeddings to get the results of the semantic search close to the current query,

""")

with tab3:
    st.markdown("# Demonstration")
    query = st.text_input('Ask a question to the LLM')
    submit_btn =  st.button('Submit')
    if submit_btn:
        with st.spinner("Processing..."):
            response= submit_query(query=query,useVectorStore=callWithVectorStore)
            st.write(response)

with st.sidebar:

    selected_model = st.selectbox("Possible LLM", ["Claude", "OpenAI"])
    
    callWithVectorStore = st.toggle("Activate Vector Store")
    
    pdf = st.file_uploader('Upload your Document', type='pdf')
    if pdf is not None:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        kb = process_text(text)

        # Text variable will store the pdf text
        
        query = st.text_input('Ask a question to the PDF')
        cancel_button = st.button('Cancel')
        
        
        
        if query:
            llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

            <context>
            {context}
            </context>

            Question: {input}""")

            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = kb.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": query})
            st.write(response["answer"])

