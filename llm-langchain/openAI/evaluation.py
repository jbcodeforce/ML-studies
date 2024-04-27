"""
Evaluate the quality of the answer for a Retriever.
Need to get the gen ai domain vector store.
It uses the QAGenerateChain
"""
import sys, os
import langchain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.evaluation.qa import QAGenerateChain

from dotenv import load_dotenv
load_dotenv(dotenv_path="../../.env")

embeddings = OpenAIEmbeddings()
CHROMA_DB_FOLDER="./agent_domain_store/"

vectorstore=None

if os.path.isabs(CHROMA_DB_FOLDER):
    print("use the build_agent_domain script to create the vector store")
    sys.exit(1)
else:
    vectorstore=Chroma(persist_directory=CHROMA_DB_FOLDER,embedding_function=embeddings)


llm = ChatOpenAI(temperature = 0.0)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

print(vectorstore.get(["1","2"]))
example_gen_chain = QAGenerateChain.from_llm(llm)
examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in vectorstore.get(["1","2","3","4"])]
)
print(examples)
langchain.debug = False
sys.exit()
rep=qa.run(examples[0]["qa_pairs"]["query"])

"""
>>> The trace will have each step of the chain presented:

[chain/start] [1:chain:RetrievalQA] Entering Chain run with input:
{
  "query": "What is the term used .... in a database?"
}
[chain/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain] Entering Chain run with input:
[inputs]
[chain/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain] Entering Chain run with input:
{
  "question": "What is the term used ...  database?",
  "context": "Fig. 8. Categorization of human memory....\

>>> Here the context is the retrieved doc chunks. The the chain goes to the llm

[llm/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain > 5:llm:ChatOpenAI] 
Entering LLM run with input:
{
  "prompts": [
    "System: Use the following pieces of context to answer the user's question. 

>>> getting out of the LLM calls, it returns structured answer:

[llm/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain > 5:llm:ChatOpenAI] [969ms] 
Exiting LLM run with output:
{
  "generations": [
    [
      {
        "text": "The term used to refer to a unique identifier .. called a \"primary key.\"",
        "generation_info": {
          "finish_reason": "stop",
          "logprobs": null
        },
        ...
 "llm_output": {
    "token_usage": {
      "completion_tokens": 25,
      "prompt_tokens": 576,
      "total_tokens": 601
    },


"""
print(rep)

# To evaluate the questions and answers with a predefined chain:
from langchain.evaluation.qa import QAEvalChain
examples_as_list = [ e["qa_pairs"] for e in examples]
predictions = qa.batch(examples_as_list)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples_as_list):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()