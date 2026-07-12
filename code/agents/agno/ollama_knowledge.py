from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from agno.db.sqlite.sqlite import SqliteDb
from agno.models.ollama import Ollama

contents_db = SqliteDb(db_file="data/contents.db",
        knowledge_table= "flink-km"
        )

vector_db = ChromaDb(
    name="test_vector_db",
    collection="test_collection",
    path="data/chroma",
)



def create_sync_knowledge() -> Knowledge:
    return Knowledge(
        name="My AI Assistant Knowledge Base",
        description="Flink Knowledge Implementation",
        vector_db=vector_db,
        contents_db=contents_db,
    )

instructions = """\
You are an expert on the Apache Flink framework and building AI event-driven agents.

## Workflow

1. Search
   - For questions about Flink, always search your knowledge base first
   - Extract key concepts from the query to search effectively

2. Synthesize
   - Combine information from multiple search results
   - Prioritize official documentation over general knowledge

3. Present
   - Lead with a direct answer
   - Include code examples when helpful
   - Keep it practical and actionable

## Rules

- Always search knowledge before answering Agno questions
- If the answer isn't in the knowledge base, say so
- Include code snippets for implementation questions
- Be concise — developers want answers, not essays\
"""

agent_db = SqliteDb(db_file="tmp/agents.db")

if __name__ == "__main__":
    url = "https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/table/tuning/"
    knowledge = create_sync_knowledge()
    knowledge.insert(name= "flink-performance-tuning",
        url=url,
        metadata= {"user_tags": "flink_performance"})
    agent = Agent(
        name="My AI Assistant",
        instructions=instructions,
        description="My AI Assistant",
        model=Ollama(id="mistral:7b-instruct"),
        knowledge=knowledge,
        search_knowledge=True,
        debug_mode=True,
        markdown=True,
        num_history_runs=5,
        add_history_to_context=True,
        db=agent_db
    )
    agent.print_response(
        "What is flink minibatch aggregation?",
        markdown=True,
    )
    knowledge.remove_vectors_by_name("flink-performance-tuning")