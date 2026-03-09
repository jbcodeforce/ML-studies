"""
Hunggingface embeddings
support splitting a markdown file and create embedding for each chunks

Use sentence transformers https://www.sbert.net/. 

Then we can use FAISS to do semantic search
"""
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pickle,os
import faiss

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def build_splits_from_md_doc(path):
    """
    return a list of Documents of this form
    Document(page_content="...", 
            metadata={'Header 1': 'D...p', 'Header 2': 'Fam...s'})
    """
    with open(path, 'r') as file:
        markdown_document = file.read()
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_document)
        chunk_size = 90
        chunk_overlap = 20
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(md_header_splits)
        

def build_embedding(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(docs)

if __name__ == "__main__":
    sentences = []
    embeddings = []
    if not os.path.exists("embeddings.pkl"): 
        print(" ---- loading file and create embeddings")
        documents = build_splits_from_md_doc("./example.md")
       
        sentences= [ d.page_content for d in documents]
        embeddings = build_embedding(sentences)
        print(embeddings.shape)
        # with Langchain Faiss vector store  we need to keep the page_content
        with open("embeddings.pkl", "wb") as fOut:
            # pickle.dump({"documents": documents, "embeddings": embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump({"documents": sentences, "embeddings": embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("--- load previous embedding")
        with open("embeddings.pkl", "rb") as fIn:
            stored_data = pickle.load(fIn)
            documents = stored_data["documents"]
            embeddings = stored_data["embeddings"]
    d=embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings) 
    print(index.ntotal)
    queryText="what are the issues with Dave Grahan?"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    xq = model.encode([queryText])
    D, I = index.search(xq,k=4)
    print(D)
    print(I)

    for i in I[0]:
        print(stored_data["documents"][i])
    