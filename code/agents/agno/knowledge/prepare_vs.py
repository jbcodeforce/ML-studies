"""
This script prepares the knowledge base for the agent.
"""
import os
import agno
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.search import SearchType
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from datetime import datetime
from pydantic import BaseModel
from typing import List
import json
import hashlib
from dotenv import load_dotenv
_env_file = os.getenv("ML_ENV_FILE")
load_dotenv(_env_file) if _env_file else load_dotenv()
# Default URL (e.g. mlx-llm-server or OpenAI-compatible proxy on 1337)
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:7999/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Ornith-1.0-9B-6bit")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "local_key")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "astronomy"

class DocumentMetadata(BaseModel):
    name: str
    path: str
    url: str
    src_type: str
    sha256: str
    processed: bool
    updated_at: str
    metadata: dict

def prepare_knowledge_base() -> Knowledge:
    """
    Prepare the knowledge base for the agent.
    """
    embedder = FastEmbedEmbedder()

    vector_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        search_type=SearchType.hybrid,
        embedder=embedder,
    )
    knowledge_base = Knowledge(
        vector_db=vector_db,
    )
    return knowledge_base


def load_documents_from_manifest(manifest_file: str) -> List[DocumentMetadata]:
    """
    Load document metadata from a manifest file.
    """
    with open(manifest_file, "r") as f:
        data = json.load(f)
    return [DocumentMetadata.model_validate(item) for item in data]

def update_manifest(manifest_file, document: DocumentMetadata, absolute_path: str, manifests: List[DocumentMetadata]):
    """
    Update the manifest file with the document metadata.
    """
    document.processed = True
    document.updated_at = datetime.now().isoformat()
    document.sha256 = hashlib.sha256(open(absolute_path, "r").read().encode()).hexdigest()
    with open(manifest_file, "w") as f:
        json.dump([document.model_dump() for document in manifests], f)


def process_manifests(kb: Knowledge, current_dir: str):
    """
    Process the manifest file.
    """

    manifest_file = os.path.join(current_dir, "test_docs", "manifest.json")
    print(f"Process manifest file: {manifest_file}")
    manifest = load_documents_from_manifest(manifest_file)
    if manifest is None:
        return
    for document in manifest:
        if document.processed:
            continue
        absolute_path = str(current_dir) + '/test_docs/' + document.path
        print(f"Process document: {document.name} in {absolute_path} with metadata: {document.metadata}")
        kb.insert(name=document.name, 
                  path=absolute_path, 
                  metadata=document.metadata)
        update_manifest(manifest_file, document, absolute_path, manifest)

if __name__ == "__main__":
    kb = prepare_knowledge_base()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    process_manifests(kb, current_dir)
    #kb.vector_db.delete_by_name("Introduction")
    