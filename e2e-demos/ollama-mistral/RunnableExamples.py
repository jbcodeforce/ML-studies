from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import functools



def simple_runnable():
    runnable = RunnableParallel(
    passed = RunnablePassthrough(),
    extra= RunnablePassthrough.assign(mult= lambda x:x["num"] * 3),
    modified=lambda x:x["num"] +1   
    )

    print(runnable.invoke({"num": 6}))
    

def simple_chain_to_mistral():
    llm = ChatOllama(model="mistral")
    prompt = ChatPromptTemplate.from_messages(
        [
            ( "system", """
            write out 2 flashcard questions with their options based on the given topic below.
            Use the format\nQUESTIONS:...\nOPTIONS:...\nSOLUTION:...\n\n
            """),
            ("human", "{topic}"),
        ]
    )

    runnable= (
        {"topic": RunnablePassthrough()} | prompt | llm | StrOutputParser()
            )

    print(runnable.invoke("the cuban missile crisis"))
    print("Using binding to stop before the first occurrence of the solution so the question has no solution")

    runnable= (
        {"topic": RunnablePassthrough()} | prompt | llm.bind(stop="SOLUTION") | StrOutputParser()
            )
    print(runnable.invoke("the cuban missile crisis"))

def return_questions(questions: list[str]):
    for q in questions:
        return q + "\n"
    
def tool_calling_with_mistral():
    tools = {
            "name": "return_questions",
            "description": "Extract questions from raw text",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_text": {
                        "type": "string",
                        "description": "The raw text of a flashcard questions",
                    },
                    "questions": {
                        "type": "array",
                        "description": "array of questions",
                        "items": {
                            "type": "string",
                            "description": "question string"
                        }
                    }
                },
                
                "required": ["questions", "solution"],
            },
        }
    prompt = ChatPromptTemplate.from_messages(
        [
            ( "system", """
            write out flashcard questions to the following topic below then return the list of questions.
            """),
            ("human", "{topic}"),
        ]
    )
    names_of_functions = { "return_questions": functools.partial(return_questions)}
    llm = ChatOllama(model="mistral").bind(
        function_call={"name": "return_questions"}, functions = [tools]
    )
    runnable= (
        {"topic": RunnablePassthrough()} | prompt | llm 
            )
    print(runnable.invoke("the cuban missile crisis"))
    
if __name__ == "__main__":
    #simple_runnable()
    #simple_chain_to_mistral()
    tool_calling_with_mistral()