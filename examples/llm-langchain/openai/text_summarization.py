from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load API KEY
load_dotenv(dotenv_path="../../.env")

print("--- Load a blog content with beautiful soup")
URL="https://www.tecton.ai/blog/top-3-benefits-of-implementing-a-feature-platform/"

loader = WebBaseLoader(URL)
docs = loader.load()
content_to_summarize=docs[0].page_content

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
prompt =  ChatPromptTemplate.from_messages([
    ("system", "Use the provided articles delimited by triple quotes, and as writer expert, summarize the text"),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
resp=chain.invoke({"input": content_to_summarize})

# The results should includes some hallucinations
print(f"\n--- Summary:\n\n {resp}")