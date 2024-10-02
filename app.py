import os
from crewai import Crew, Task, Agent
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.tools import Tool
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

import asyncio
import aiohttp

from helper import ingest_and_retrieve_docs, ingest
from soundutils import initialize_audio, process_and_play_text


# Load environment variables and set up API keys
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')



if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables")

# Set up LLM
llm = ChatGroq(
    temperature=0,
    model_name="groq/llama3-groq-70b-8192-tool-use-preview",
    api_key=GROQ_API_KEY
)

# Set up tools
web_search_tool = TavilySearchResults(k=2)


query_retrieval_tool = Tool(
    name="QueryBasedRetrieval",
    func=ingest_and_retrieve_docs,
    description="Retrieves relevant document chunks based on a given query"
)


all_chunk_retrieval_tool = Tool(
    name="AllChunkRetrieval",
    func=lambda x: ingest(),  # Use a lambda function to call ingest without arguments
    description="Retrieves all document chunks from the database"
)

# Define agents
intent_classifier = Agent(
    role='Intent Classifier',
    goal='Classify the intent of the user query',
    backstory="You are an expert at understanding user intents. You classify queries into categories like 'greeting', 'document_inquiry', or 'general_question'.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

greeter = Agent(
    role='Greeter',
    goal='Provide a friendly greeting to the user',
    backstory="You are a warm and welcoming greeter, always ready to make users feel comfortable.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


summarize_agent = Agent(
    role='Document Summarizer',
    goal='Retrieve all document chunks and provide a comprehensive summary of the entire document',
    backstory="You are an expert at synthesizing information from document chunks to provide comprehensive summaries. Always use the AllChunkRetrieval tool to get the document contents before summarizing. After retrieving the chunks, analyze them thoroughly to identify main topics, key points, and the overall structure of the document.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[all_chunk_retrieval_tool]
)


context_retriever = Agent(
    role='Context Retriever',
    goal='Retrieve relevant context from the vector database',
    backstory="You are an expert at finding relevant information from the vector database. You always explain your actions and reasoning clearly.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[query_retrieval_tool]
)

context_analyzer = Agent(
    role="Context Analyzer",
    goal="Analyze if the retrieved context is sufficient to answer the question",
    backstory="You are skilled at determining if the given context can fully answer a question. You always explain your analysis process and decision-making clearly.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

web_searcher = Agent(
    role="Web Searcher",
    goal="Search the web for information about the original question when needed",
    backstory="You are proficient at finding relevant information on the web, always focusing on the original question asked. You explain your search strategy and why you chose specific results.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[web_search_tool]
)

answer_formulator = Agent(
    role='Answer Formulator',
    goal='Formulate a comprehensive answer based on available information',
    backstory="You are expert at crafting clear and concise answers from given information, always ensuring the answer relates to the original question. You explain your thought process in formulating the answer and highlight key points from the sources used.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define tasks
def create_intent_classification_task(query):
    return Task(
        description=f"Classify the intent of the query: '{query}'. Determine if it's a greeting, a document inquiry, or a general question.",
        agent=intent_classifier,
        expected_output="A string indicating the classified intent: 'greeting', 'document_inquiry', or 'general_question'"
    )

def create_greeting_task():
    return Task(
        description="Provide a warm and friendly greeting to the user.",
        agent=greeter,
        expected_output="A friendly greeting message"
    )



def create_summarize_task():
    return Task(
        description="First, use the AllChunkRetrieval tool to get all document chunks. Then, analyze these chunks and provide a comprehensive summary of the entire document. The summary should cover the main topics, key points, and overall structure of the document.",
        agent=summarize_agent,
        expected_output="A comprehensive summary of the document contents, including main topics, key points, and overall structure."
    )

def create_retrieval_task(original_question):
    return Task(
        description=f"Use the QueryBasedRetrieval tool to find relevant context for the question: '{original_question}'.",
        agent=context_retriever,
        expected_output="Retrieved context relevant to the question"
    )

def create_analysis_task(original_question):
    return Task(
        description=f"Analyze if the retrieved context is sufficient to answer the question: '{original_question}'.",
        agent=context_analyzer,
        expected_output="Analysis of context sufficiency: 'sufficient' or 'insufficient'"
    )

def create_web_search_task(original_question):
    return Task(
        description=f"If the context is insufficient, perform a web search to find information about the question: '{original_question}'.",
        agent=web_searcher,
        expected_output="Additional information from web search, if needed"
    )

def create_answer_task(original_question):
    return Task(
        description=f"Formulate a comprehensive answer to the question: '{original_question}' based on the available information.",
        agent=answer_formulator,
        expected_output="A comprehensive answer to the original question"
    )

# Define crews
def create_intent_crew():
    return Crew(
        agents=[intent_classifier],
        tasks=[create_intent_classification_task("placeholder")],
        verbose=True
    )

def create_greeting_crew():
    return Crew(
        agents=[greeter],
        tasks=[create_greeting_task()],
        verbose=True
    )

def create_document_summary_crew():
    return Crew(
        agents=[summarize_agent],
        tasks=[create_summarize_task()],
        verbose=True
    )

def create_qa_crew(question):
    return Crew(
        agents=[context_retriever, context_analyzer, web_searcher, answer_formulator],
        tasks=[
            create_retrieval_task(question),
            create_analysis_task(question),
            create_web_search_task(question),
            create_answer_task(question)
        ],
        verbose=True
    )

# Main function to process queries
# def process_query(query):
#     intent_crew = create_intent_crew()
#     intent_crew.tasks[0].description = f"Classify the intent of the query: '{query}'. Determine if it's a greeting, a document inquiry, or a general question."
#     x = intent_crew.kickoff()
#     intent = str(x)


#     if "greeting" in intent:
#         greeting_crew = create_greeting_crew()
#         return greeting_crew.kickoff()
#     elif "document_inquiry" in intent:
#         document_crew = create_document_summary_crew()
#         return document_crew.kickoff()
#     else:
#         qa_crew = create_qa_crew(query)
#         return qa_crew.kickoff()

# # Example usage
# if __name__ == "__main__":
#     user_query = "Hi I am Asheesh"
#     result = process_query(user_query)
#     print(f"Final Answer: {result}")


import streamlit as st

async def process_query(query):
    intent_crew = create_intent_crew()
    intent_crew.tasks[0].description = f"Classify the intent of the query: '{query}'. Determine if it's a greeting, a document inquiry, or a general question."
    intent_result = await asyncio.to_thread(intent_crew.kickoff)
    intent = str(intent_result)

    if "greeting" in intent:
        greeting_crew = create_greeting_crew()
        result = await asyncio.to_thread(greeting_crew.kickoff)
        tasks = greeting_crew.tasks
    elif "document_inquiry" in intent:
        document_crew = create_document_summary_crew()
        result = await asyncio.to_thread(document_crew.kickoff)
        tasks = document_crew.tasks
    else:
        qa_crew = create_qa_crew(query)
        result = await asyncio.to_thread(qa_crew.kickoff)
        tasks = qa_crew.tasks

    return result, tasks

async def main(query):
    initialize_audio()
    async with aiohttp.ClientSession() as session:
        result, tasks = await process_query(query)
        # st.write(f"Final Answer: {result}")
        
        for i, task in enumerate(tasks):
            st.write(f"\nReasoning {i+1} Output:")
            output = str(task.output)
            st.write(f"Result: {output}")
            
            try:
                await process_and_play_text(session, output)
            except Exception as e:
                st.error(f"An error occurred during text-to-speech processing: {str(e)}")
        st.write(f"Final Answer: {result}")
    st.success("All tasks processed and played.")

def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

st.title("Query Processing and Audio Playback App")

user_query = st.text_input("Enter your query:")

if st.button("Process Query"):
    if user_query:
        run_async(main(user_query))
    else:
        st.warning("Please enter a query.")

if __name__ == "__main__":
    st.write("App is running.")