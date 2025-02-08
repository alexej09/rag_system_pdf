# RAG System for PDF Documents with Open Source Models

This repository provides an end-to-end **Retrieval-Augmented Generation (RAG)** system for processing and querying PDF documents using **open-source AI models**. The system leverages **LangChain**, **Ollama**, and **ChromaDB** for vector storage while utilizing **OCR libraries** for document preprocessing.

## Features
- **Extract text from PDFs** using open-source OCR tools
- **Chunk and vectorize document contents** for efficient retrieval
- **Store embeddings** in **ChromaDB**, an open-source vector database
- **Query documents** using **LangChain’s retrievers and language models**
- **Use Ollama to run open-source models like Llama 3**

## Architecture
1. **Preprocessing PDFs**:
   - Parses PDF content using **Nougat** OCR.
   - Splits the extracted text into **overlapping chunks** for better retrieval.
2. **Creating Embeddings and Vector Storage**:
   - Uses **nomic-embed-text** to generate text embeddings.
   - Stores embeddings in **ChromaDB** for fast retrieval.
3. **Retrieval-Augmented Generation (RAG)**:
   - Uses **LangChain’s MultiQueryRetriever** to refine search queries.
   - Generates responses using **Ollama’s open-source models** like **Llama3.2**.
4. **Interactive Chat Interface**:
   - Enables users to ask questions about the processed documents.
   - Fetches relevant context from the vector store before generating answers.

## Requirements
Ensure you have the following installed:
- Python 3.10+
- [Ollama](https://ollama.ai/) install for running LLMs locally
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/de/downloads/) install for running langchain tools
- Install dependencies:
```sh
  pip install -r requirements.txt
```
## Usage
To process a PDF file and store its embeddings in a vector database, execute:
```sh
python vector_db_nougat.py
```
To read the content of the vector database, use:
```sh
python read_vector_db.py
```
To chat with the database, run following script:
```sh
chat_with_vector_db.py
```

## Tools and Libraries
- **LangChain** ([Docs](https://python.langchain.com/)) - Framework for LLM-powered applications
- **Ollama** ([Website](https://ollama.ai/)) - Open-source LLM serving
- **ChromaDB** ([Docs](https://www.trychroma.com/)) - Open-source vector database
- **Nougat OCR** ([Repo](https://github.com/facebookresearch/nougat)) - OCR library for academic PDFs
- **nomic-embed-text** ([Docs](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)) - Embedding model

## Example Usage
After running the chatbot script, you can ask questions based on the uploaded PDFs. Example:
```
Your question: What are the key findings in this research paper?
```
The system will return the most relevant answer based on the extracted document content.

You can interact with the chatbot in the terminal by entering your questions as a chat. To exit the script, simply type `exit` in the terminal.

## License
This project is open-source and available under the MIT License.

## Additional Knowledge Base
- **Nougat OCR**: [YouTube Video](https://www.youtube.com/watch?v=XKBU7ROKjaQ&t=5s)
- **Mindee docTR**: [YouTube Video](https://www.youtube.com/watch?v=3nYPIDCToes)
