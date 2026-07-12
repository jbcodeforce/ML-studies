"""RAG (Retrieval-Augmented Generation) utilities."""

from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    add_start_index: bool = True,
) -> RecursiveCharacterTextSplitter:
    """Create a configured text splitter for document chunking.
    
    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        add_start_index: Whether to add start index to metadata
        
    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )


def split_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into smaller chunks.
    
    Args:
        docs: List of documents to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of split document chunks
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs)


def create_chroma_vectorstore(
    docs: list[Document],
    embedding: Any,
    persist_directory: str | None = None,
    collection_name: str = "default",
):
    """Create a Chroma vector store from documents.
    
    Args:
        docs: List of documents to index
        embedding: Embedding model to use
        persist_directory: Optional directory to persist the store
        collection_name: Name of the collection
        
    Returns:
        Chroma vector store instance
    """
    from langchain_chroma import Chroma
    
    kwargs = {
        "documents": docs,
        "embedding": embedding,
        "collection_name": collection_name,
    }
    if persist_directory:
        kwargs["persist_directory"] = persist_directory
        
    return Chroma.from_documents(**kwargs)


def create_retriever(
    vectorstore: Any,
    search_type: str = "similarity",
    k: int = 6,
):
    """Create a retriever from a vector store.
    
    Args:
        vectorstore: Vector store instance
        search_type: Type of search (similarity, mmr, etc.)
        k: Number of documents to retrieve
        
    Returns:
        Configured retriever
    """
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

