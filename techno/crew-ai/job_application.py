from crewai import Crew, Task, Agent
from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool
from dotenv import load_dotenv
import os, getopt,sys

load_dotenv("../../.env")

SERPER_API_KEY= os.getenv("SERPER_API_KEY")
"""
A program using multiple agents to analyse a job posting, do personal profiling, and resume tuning
from a source resume to taylor it for the job
Use web scrapping and search tools.
"""
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path='./a_resume.md')
semantic_search_resume = MDXSearchTool(mdx='./a_resume.md')

researcher = Agent(
    role="Tech Job Researcher",
    goal="Make sure to do amazing analysis on "
         "job posting to help job applicants",
    tools = [scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Job Researcher, your process in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
    )
)

profiler = Agent(
    role="Personal Profiler for Engineers",
    goal="Do incredible research on job applicants "
         "to help them stand out in the job market",
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "Equipped with analytical process, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    )
)

resume_strategist = Agent(
    role="Resume Strategist for Engineers",
    goal="Find all the best ways to make a "
         "resume stand out in the job market.",
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    )
)

interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Create interview questions and talking points "
         "based on the resume and job requirements",
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    )
)

research_task = Task(
    description=(
        "Analyze the job posting URL provided ({job_posting_url}) "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    async_execution=True
)

profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the GitHub ({github_url}) URLs, and personal write-up "
        "({personal_writeup}). Utilize tools to extract and "
        "synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    async_execution=True
)

resume_strategy_task = Task(
    description=(
        "Using the profile and job requirements obtained from "
        "previous tasks, tailor the resume to highlight the most "
        "relevant areas. Employ tools to adjust and enhance the "
        "resume content. Make sure this is the best resume even but "
        "don't make up any information. Update every section, "
        "inlcuding the initial summary, work experience, skills, "
        "and education. All to better reflrect the candidates "
        "abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated resume that effectively highlights the candidate's "
        "qualifications and experiences relevant to the job."
    ),
    output_file="tailored_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist
)

interview_preparation_task = Task(
    description=(
        "Create a set of potential interview questions and talking "
        "points based on the tailored resume and job requirements. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candiadte highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
    output_file="interview_materials.md",
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer
)

job_application_crew = Crew(
    agents=[researcher,
            profiler,
            resume_strategist,
            interview_preparer],

    tasks=[research_task,
           profile_task,
           resume_strategy_task,
           interview_preparation_task],

    verbose=True
)





def usage():
    print("--- Job application resume tayloring ---")
    print("python3 job_application -u URL_OF_JOB -g GITHUB_URL -p personal_write_up")

def parse_args():
    GITHUB='https://github.com/jbcodeforce'
    PERSONAL="""Jerome is an accomplished Software
        Engineering Leader with 25 years of experience, former ibm distinguished engineer, 
        and principal solution architect at AWS.
    """
    URL=""
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hu:g:p:",["url=","github=","pers_write_up="])
    except getopt.GetoptError:
        print(usage())
        sys.exit(2)
    for opt, arg in opts:
        print(opt, arg)
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-u", "--url"):
            URL = arg
        elif opt in ("-g", "--github"):
            GITHUB = arg
        elif opt in ("-p", "--pers_write_up"):
            PERSONAL = arg
    return URL,GITHUB,PERSONAL

if __name__ == "__main__":    
    URL,GITHUB,PERSONAL = parse_args()
    job_application_inputs = {
        'job_posting_url': URL,
        'github_url': GITHUB,
        'personal_writeup': PERSONAL
    }
    print(job_application_inputs)
    
    result = job_application_crew.kickoff(inputs=job_application_inputs)
    print(result)

