from datasets import (
    load_dataset,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Cassandra
from cassandra.cluster import Cluster
import cassio

from dotenv import load_dotenv

load_dotenv("../../.env")
CASSANDRA_KEYSPACE="owlkeyspace"

embe = OpenAIEmbeddings()
cluster = Cluster(["127.0.0.1"])
session = cluster.connect()
cassio.init(session=session, keyspace=CASSANDRA_KEYSPACE)
vstore = Cassandra(
    embedding=embe,
    table_name="cassandra_vector_demo",
)

philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]

docs = []
for entry in philo_dataset:
    metadata = {"author": entry["author"]}
    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)

inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")

results = vstore.similarity_search("Our life is what we make of it", k=3)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")