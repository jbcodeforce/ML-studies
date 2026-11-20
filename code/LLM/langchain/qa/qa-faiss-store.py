import os
import sys
from langchain.prompts import PromptTemplate
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA

# SECURITY FIX: Avoid sys.path.append which can cause module injection attacks.
# Use package imports or proper module resolution instead.
# The bedrock.utils module should be installed as a proper package, not
# loaded via path manipulation.
#
# If bedrock.utils is not available as an installed module, consider:
# 1. Installing it properly: pip install bedrock-utils (or equivalent)
# 2. Creating a proper package structure with __init__.py
# 3. Using relative imports within the package

try:
    # Try standard import first
    from bedrock.utils import bedrock, print_ww
except ImportError:
    # Fallback: import from the project's bedrock package
    # This is safer than sys.path.append
    import importlib.util
    module_path = os.path.join(os.path.dirname(__file__), '..', 'bedrock', 'utils.py')
    spec = importlib.util.spec_from_file_location("bedrock.utils", module_path)
    bedrock_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bedrock_utils)
    bedrock = bedrock_utils.bedrock
    print_ww = bedrock_utils.print_ww

bedrock_runtime = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock_runtime, model_kwargs={'max_tokens_to_sample':200})
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)

db = FAISS.load_local("faiss_index", embeddings)



prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

query = """Is it possible that I get sentenced to jail due to failure in filings?"""
answer = qa({"query": query})
print_ww(answer)

query_2 = "What is the difference between market discount and qualified stated interest"
answer_2  =answer = qa({"query": query_2})
print_ww(answer_2)
