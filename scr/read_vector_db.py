# import os
import logging
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "rag_papers"
PERSIST_DIRECTORY = "./data/processed/chroma_db"

# Chroma-Instanz erstellen
vector_db = Chroma(persist_directory=PERSIST_DIRECTORY)

# Verfügbare Sammlungen abrufen und ausgeben
collections = vector_db._client.list_collections()
print("Verfügbare Sammlungen:")
for collection in collections:
    print(collection)

# Sammlung laden
collection = vector_db._client.get_collection(VECTOR_STORE_NAME)
documents = collection.get(include=["documents", "metadatas", "embeddings"])
print(f"Die Sammlung '{VECTOR_STORE_NAME}' enthält {collection.count()} Chunks.")

# Chuncks von Dokumenten, Metadaten und Embeddings durchgehen
for i, (doc, metadata, embedding) in enumerate(zip(documents["documents"], documents["metadatas"], documents["embeddings"])):
    print(f"Dokument_Chunck {i}: {doc[:1000]}")
    # print(f"Metadata_Chunck {i}: {metadata[:1000]}")
    print(f"Embedding_Chunck {i}: {str(embedding)[:1000]}")