"""
streamlit app to upload a pdf and then ask questions against it
"""
import os,base64
import streamlit as st
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv("../../.env")

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
    return FAISS.from_texts(chunks, embeddings)


def view_app():
    st.title("Chat with your PDF 💬")
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    if pdf is not None:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        kb = process_text(text)

        # Text variable will store the pdf text
        
        query = st.text_input('Ask a question to the PDF')
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
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

if __name__ == "__main__":
    view_app()