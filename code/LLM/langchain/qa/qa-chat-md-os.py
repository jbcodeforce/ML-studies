'''
A very simple chat bot using Claude LLM hosted on AWS bedrock with my eda markdown files.
It uses open-search for vector store.
'''
import os, logging
import sys
from pathlib import Path
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")

def loadSourceDocumentsAsChunks(srcdir):
    '''
    Load a source directory with all md files in it. and split it into chunks.
    
    SECURITY: Validates input directory path to prevent directory traversal attacks.
    '''
    # Normalize and resolve the path
    resolved_path = Path(srcdir).resolve()
    
    # Prevent directory traversal by ensuring the resolved path 
    # is within the project root
    project_root = Path(__file__).resolve().parents[4]  # 4 levels up from code/LLM/langchain/qa/
    if not resolved_path.is_relative_to(project_root):
        logger.warning(
            f"Path traversal attempted: {resolved_path}. "
            f"Accessing project-relative path instead."
        )
        # Fallback to a safe project-relative path
        resolved_path = project_root / "docs"
    
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {resolved_path}")
    
    logger.info("--- Load document, split ---")
    loader = DirectoryLoader(str(resolved_path), glob="**/*.md")
    documents = loader.load()  # text and metadata
    logger.debug(f"Loaded {len(documents)} documents")
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    all_splits = []
    for document in documents:
        all_splits += markdown_splitter.split_text(document)
    return all_splits

if __name__ == "__main__":
    splits = loadSourceDocumentsAsChunks("../docs")
    print(f"Total splits: {len(splits)}")
