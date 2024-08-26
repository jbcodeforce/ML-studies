import getopt, sys
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
load_dotenv("../../.env")

DOMAIN_VS_PATH="./agent_vstore"
resume_vectorstore = None

def process_file(filename):
    _,type= filename.split(".")
    if type == "pdf":
        splits= split_pdf_docs(filename)
    elif type == "md":
        splits= split_md_docs(filename)
    else:
        print("ERROR file type not yet supported")
        sys.exit(2)
    return splits
    

def save_to_vectore_store(documents, collection_name):
    global resume_vectorstore
    resume_vectorstore =  Chroma.from_documents(documents=documents, 
                                         embedding=OpenAIEmbeddings(),
                                         collection_name=collection_name,
                                         persist_directory=DOMAIN_VS_PATH)
    


def split_pdf_docs(file_uri: str) -> List[Document]:
    loader = PyMuPDFLoader(file_uri)
    documents = loader.load_and_split()
    return documents

def split_md_docs(file_uri: str) -> List[Document]:
    """
        split a markdown file by a specified set of headers.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ]
    with open(file_uri, 'r') as file:
        markdown_document = file.read()
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(markdown_document)
        chunk_size = 250
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                       chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(md_header_splits)
        return splits

def read_resume(filename, collection_name) -> List[str]:
    splits=process_file(filename)
    save_to_vectore_store(splits, collection_name)
    content=[]
    for doc in splits:
        content.append(doc.page_content)
    return content

def extract_job_requirements_agent(text) -> str:
    prompt=ChatPromptTemplate.from_messages([
                        ("system", """Analyze the job posting provided as: \n {job_application}\n\n to extract key skills, 
                                    experiences, and qualifications required. Gather content and identify and categorize the requirements.
                                    The expected_output is a structured list in markdown format of job requirements, including necessary skills, 
                                    qualifications, and experiences."""),
                        MessagesPlaceholder(variable_name="job_application", optional=True)
                    ])
    llm = ChatOpenAI()
    chain = prompt | llm
    ai_message = chain.invoke({"job_application" : [text]})
    lines=ai_message.content.replace("\\n", '\n').splitlines()
    return lines

def resume_profiler_agent(resume_sentences):
    prompt=ChatPromptTemplate.from_messages([
                        ("system", """You are a personal Profiler for Engineers. Your goal is to do incredible research on job applicants
                            to help them stand out in the job market. \n
                         Compile a detailed personal and professional profile using the personal resume: {personal_writeup}.\n 
                         The expected output is a comprehensive profile document that includes skills, project experiences, contributions, 
                         interests, and communication style."""),
                        MessagesPlaceholder(variable_name="personal_writeup", optional=True)
                    ])
    llm = ChatOpenAI()
    chain = prompt | llm
    ai_message=chain.invoke({"personal_writeup" : resume_sentences})
    lines=ai_message.content.replace("\\n", '\n').splitlines()
    return lines

def resume_strategist_agent(candidate_profile, job_needs):
    prompt=ChatPromptTemplate.from_messages([
                        ("system", """You are a resume strategist for Engineers. Your goal is to find all the best ways to make a resume stand out in the job market.
                         \n
                         With a strategic mind and an eye for detail, you excel at refining resumes to highlight the most
                        relevant skills and experiences, ensuring they resonate perfectly with the job's requirements. \n\n
                         
                        Using the profile in {{profile}} and the job requirements {{job_requirements}} 
                        tailor the resume to highlight the most relevant areas. Adjust and enhance the 
                        resume content. Make sure this is the best resume even but don't make up any information. Update every section, 
                        including the initial summary, work experience, skills, and education. All to better reflect the candidate
                        abilities and how it matches the job posting.
                         """),
                        MessagesPlaceholder(variable_name="profile", optional=True),
                        MessagesPlaceholder(variable_name="job_requirements", optional=True),
                    ])
    llm = ChatOpenAI()
    chain = prompt | llm
    ai_message= chain.invoke({"profile" : candidate_profile,
                         "job_requirements": job_needs})
    lines=ai_message.content.replace("\\n", '\n').splitlines()
    return lines

def read_application(filename):
    """ read the job application content and extract the job requirements
    """
    with open(filename, "r") as f:
        text=f.read()
        return extract_job_requirements_agent(text)

def usage():
    print("--- Job application resume tayloring ---")
    print("python3 main -a md_file_for_app  -r current_resume")

def parse_args():
    JOB_APPLICATION="job_app_1.md"
    RESUME=""
    COLLECTION_NAME="profiles"
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ha:r:c:",["application=","resume=", "collection="])
    except getopt.GetoptError:
        print(usage())
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-a", "--application"):
            JOB_APPLICATION = arg
        elif opt in ("-r", "--resume"):
            RESUME= arg
        elif opt in ("-c", "--collection"):
            COLLECTION_NAME= arg
    return JOB_APPLICATION,RESUME, COLLECTION_NAME

if __name__ == "__main__":    
    JOB_APPLICATION,RESUME,COLLECTION_NAME = parse_args()
    print("\n\t1/ read and process resume\n")
    #splits=process_file(RESUME)
    resume_content=read_resume(RESUME, COLLECTION_NAME)
    print(resume_content)
    print("#"*40 + "\n")
    print("\n\t2/ read and process the job application using agent\n")
    needs=read_application(JOB_APPLICATION)
    print(needs)
    print("#"*40 + "\n")
    print("\n\t3/ profile the resume with agent\n")
    resume_profile = resume_profiler_agent(resume_content)
    print(resume_profile)
    print("#"*40 + "\n")
    print("\n\t4/ build a new resume with agent \n")
    new_resume=resume_strategist_agent(resume_profile,needs)
    print(new_resume)
    print("#"*40 + "\n")
    with open("new_resume.md","w") as f:
        for item in new_resume:
	        f.write(item+"\n")
 

    