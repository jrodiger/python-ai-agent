import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CV file is stored in the data directory
CV_PATH = Path("data") / "cv.txt"

@tool("resume_rag_tool")
def resume_rag_tool(input: str) -> str:
    """
    Searches and retrieves relevant information from the user's resume (cv.txt)
    to answer questions about their experience, skills, or education.
    Use this tool specifically for questions related to the content of the resume.
    """
    print(f"--- Using RAG Tool for query: {input} ---")
    if not CV_PATH.exists():
        return f"Error: Resume file not found at {CV_PATH}"

    try:
        loader = TextLoader(CV_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        # Initialize Ollama embeddings (ensure Ollama service is running)
        embeddings = OllamaEmbeddings(model="nomic-embed-text") # Adjust model if needed

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

        relevant_docs = retriever.invoke(input)

        # Format the retrieved documents with clear instructions
        context_parts = [doc.page_content for doc in relevant_docs]
        formatted_response = (
            "### Retrieved Resume Context\n"
            "Below are the relevant sections found in the resume. Use this information to answer the user's question:\n\n"
            + "\n---\n".join(context_parts)
            + "\n\n### Instructions\n"
            "Using ONLY the information provided above from the resume, please answer the user's question. "
            "If the information is not found in the provided context, state that clearly."
        )

        if not context_parts:
            return "No relevant information found in the resume for that query."

        return formatted_response

    except Exception as e:
        # Provide more specific error feedback if possible
        print(f"Error in RAG tool: {e}")
        # Consider returning a more informative error message to the agent/user
        return f"Error processing resume information: {e}"
