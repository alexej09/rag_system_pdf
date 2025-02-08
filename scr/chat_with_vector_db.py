import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for model and database configuration
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "rag_papers"
PERSIST_DIRECTORY = "./data/processed/chroma_db"

def load_vector_db(persist_directory, collection_name):
    """Load the existing vector database using the provided embedding model."""
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        embedding_function=embedding,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    logging.info("Existing vector database loaded.")
    return vector_db

def create_retriever(vector_db, llm):
    """Create a multi-query retriever that generates alternative questions for better search."""
    # Define a prompt to generate five different versions of the user's question
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    """Create the chain that links the retriever, prompt, LLM, and output parser."""
    # Define the prompt template for the RAG (Retrieval-Augmented Generation) process
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # Chain the components: first, get context from retriever, then format the prompt,
    # generate the answer using the LLM, and finally parse the output as a string.
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain successfully created.")
    return chain

def main():
    # Load the vector database from disk
    vector_db = load_vector_db(PERSIST_DIRECTORY, VECTOR_STORE_NAME)

    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Create the retriever for multi-perspective document retrieval
    retriever = create_retriever(vector_db, llm)

    # Build the processing chain
    chain = create_chain(retriever, llm)

    print("Vector database successfully loaded. You can now chat with the document.")
    print("Type 'exit' to quit the program.")

    while True:
        # Get the user's question
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit']:
            print("Exiting program.")
            break
        try:
            # Invoke the chain with the user's question
            response = chain.invoke(input=question)
            print("\n**Answer:**")
            print(response)
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            print("An error occurred while processing your request.")

if __name__ == "__main__":
    main()
