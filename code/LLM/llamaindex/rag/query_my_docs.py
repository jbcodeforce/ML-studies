"""
Basic RAG on the markdown files from this project using LLamaIndex API
"""
import sys
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import MarkdownReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

COLLECTION_NAME= "ml-study-col"
load_dotenv()

def load_documents(aPath : str):
    parser = MarkdownReader()
    file_extractor: dict = {".md": parser}
    documents = SimpleDirectoryReader(
                    "aPath", file_extractor=file_extractor
                ).load_data()
    return documents

def transform(documents, vector_store):
    """
    chunking, extracting metadata, and embedding each chunk (or node object)
    """
    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter( chunk_size=512, chunk_overlap=128),
        OpenAIEmbedding()
        ],
        vector_store=vector_store
        )
    nodes = pipeline.run(documents=documents, show_progress=True)
    return nodes

def ingestion_process():
    documents= load_documents("../../docs/genAI")
    vs = connect_vector_store()
    nodes = transform(documents, vs)
   
    
def connect_vector_store():
    return  ChromaVectorStore.from_params(host='localhost', port=8005, collection_name=COLLECTION_NAME)
     
    
if __name__ == "__main__":
    #ingestion_process()
    connect_vector_store()
    print("Done")
