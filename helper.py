import os
from langchain_openai import ChatOpenAI
from crewai import Crew, Task, Agent
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.tools import Tool
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from langchain_community.tools.tavily_search import TavilySearchResults


def ingest(query=None):
    # Set up the embeddings
    embeddings = FastEmbedEmbeddings()
    
    # Define the directory where your PDF files are located
    pdf_directory = "pdfs"
    
    # Define the persistent directory for the Chroma database
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "db", "chroma_db")
    
    # Check if the database already exists
    if not os.path.exists(persistent_directory):
        # Load PDF files
        loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        
        # Create and persist the Chroma database
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persistent_directory
        )
        db.persist()
        print(f"Ingestion complete. Vector database stored in {persistent_directory}")
    else:
        # Load the existing database
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        print(f"Using existing vector database from {persistent_directory}")
    
    # Retrieve all documents from the database
    all_docs = db.get()
    
    if not all_docs['documents']:
        return "No documents found in the database."
    
    results = []
    for i, (content, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas']), 1):
        if i > 3:  # Only process the first 3 chunks
            break
        result = f"Chunk {i}:\n{content}\n"
        if metadata:
            result += f"Source: {metadata.get('source', 'Unknown')}\n"
            result += f"Page: {metadata.get('page', 'Unknown')}\n"
        results.append(result)
    
    return "\n".join(results)









# Set up document ingestion and retrieval
def ingest_and_retrieve_docs(query):
    # Set up the embeddings
    embeddings = FastEmbedEmbeddings()

    # Define the directory where your PDF files are located
    pdf_directory = "pdfs"

    # Define the persistent directory for the Chroma database
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "db", "chroma_db")

    # Check if the database already exists
    if not os.path.exists(persistent_directory):
        # Load PDF files
        loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)

        # Create and persist the Chroma database
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persistent_directory
        )
        db.persist()
        print(f"Ingestion complete. Vector database stored in {persistent_directory}")
    else:
        # Load the existing database
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        print(f"Using existing vector database from {persistent_directory}")

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.8},
    )
    relevant_docs = retriever.invoke(query)
    
    results = []
    for i, doc in enumerate(relevant_docs, 1):
        result = f"Document {i}:\n{doc.page_content}\n"
        if doc.metadata:
            result += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            result += f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        results.append(result)
    
    return "\n".join(results) if results else "No relevant documents found."



# def ingest(query=None):
#     # Set up the embeddings
#     embeddings = FastEmbedEmbeddings()

#     # Define the directory where your PDF files are located
#     pdf_directory = "pdfs"

#     # Define the persistent directory for the Chroma database
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     persistent_directory = os.path.join(current_dir, "db", "chroma_db")

#     # Check if the database already exists
#     if not os.path.exists(persistent_directory):
#         # Load PDF files
#         loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
#         documents = loader.load()

#         # Split the documents into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#         )
#         texts = text_splitter.split_documents(documents)

#         # Create and persist the Chroma database
#         db = Chroma.from_documents(
#             documents=texts,
#             embedding=embeddings,
#             persist_directory=persistent_directory
#         )
#         db.persist()
#         print(f"Ingestion complete. Vector database stored in {persistent_directory}")
#     else:
#         # Load the existing database
#         db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
#         print(f"Using existing vector database from {persistent_directory}")

#     # Retrieve all documents from the database
#     all_docs = db.get()
    
#     if not all_docs['documents']:
#         return "No documents found in the database."
    
#     results = []
#     for i, (content, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas']), 1):
#         result = f"Chunk {i}:\n{content}\n"
#         if metadata:
#             result += f"Source: {metadata.get('source', 'Unknown')}\n"
#             result += f"Page: {metadata.get('page', 'Unknown')}\n"
#         results.append(result)
    
#     return "\n".join(results)