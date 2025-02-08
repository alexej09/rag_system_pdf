import os
import subprocess
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama
from logging_config import setup_logging, log_end_time

# Initialize logging
logger = setup_logging()

# Constants
PDF_DIRECTORY = "./data/raw"  # Directory containing PDF files
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "rag_papers"
PERSIST_DIRECTORY = "./data/processed/chroma_db"

def ingest_pdfs(pdf_directory):
    """
    Parse all PDFs in the given directory using the 'nougat' subprocess,
    split the parsed text into chunks (each with metadata indicating the source file),
    and store the embeddings in a single Chroma vector database.
    """
    all_chunks = []
    
    # Get a list of PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.error(f"No PDF files found in directory: {pdf_directory}")
        return None

    # Process each PDF file
    for filename in pdf_files:
        file_path = os.path.join(pdf_directory, filename)
        logger.info(f"Processing file: {filename}")

        # Check if the file exists
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            continue

        # Parse PDF with nougat
        try:
            result = subprocess.run(
                ["nougat", "parse", file_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            parsed_text = result.stdout
            logger.info(f"File '{filename}' successfully parsed with nougat.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Nougat parsing failed for '{filename}': {e.stderr}")
            continue

        # Split the parsed text into chunks with metadata indicating the source file
        try:
            document = Document(page_content=parsed_text, metadata={"source": filename})
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
            chunks = text_splitter.split_documents([document])
            logger.info(f"File '{filename}' split into {len(chunks)} chunks.")
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error splitting document '{filename}': {e}")
            continue

    if not all_chunks:
        logger.error("No PDF files processed successfully.")
        return None

    # Create embeddings
    try:
        logger.info("Pulling embedding model...")
        ollama.pull(EMBEDDING_MODEL)
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
        logger.info("Embeddings created successfully.")
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return None

    # Create and save vector database with all chunks
    try:
        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        logger.info(f"Vector database created and saved at '{PERSIST_DIRECTORY}'.")
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        return None

    logger.info("All PDF processing completed successfully.")
    return vector_db

def main():
    ingest_pdfs(PDF_DIRECTORY)
    log_end_time()

if __name__ == "__main__":
    main()
