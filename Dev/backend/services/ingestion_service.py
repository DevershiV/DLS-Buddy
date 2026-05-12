from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import os
import pickle


# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index
faiss_index = None

# Store chunks
document_chunks = []

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_FOLDER = os.path.join(BASE_DIR, "data")

FAISS_INDEX_PATH = os.path.join(
    BASE_DIR,
    "faiss_index.bin"
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "chunks.pkl"
)

def load_documents():

    documents = []

    for file in os.listdir(DATA_FOLDER):

        if file.endswith(".docx"):

            path = os.path.join(DATA_FOLDER, file)

            loader = Docx2txtLoader(path)

            documents.extend(loader.load())

    return documents


def ingest_documents():

    global faiss_index
    global document_chunks

    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Store chunk text
    document_chunks = [
        chunk.page_content for chunk in chunks
    ]

    print(f"Created {len(document_chunks)} chunks")

    # Generate embeddings
    embeddings = embedding_model.encode(
        document_chunks
    )

    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dimension)

    faiss_index.add(embeddings)

    print("FAISS index created successfully")

    # Save FAISS index
    faiss.write_index(
        faiss_index,
        FAISS_INDEX_PATH
    )

    # Save chunks
    with open(CHUNKS_PATH, "wb") as f:

        pickle.dump(document_chunks, f)

    print("FAISS index saved successfully")

def load_vector_store():

    global faiss_index
    global document_chunks

    faiss_index = faiss.read_index(
        FAISS_INDEX_PATH
    )

    with open(CHUNKS_PATH, "rb") as f:

        document_chunks = pickle.load(f)

    print("FAISS index loaded successfully")

def search_documents(query, k=3):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = faiss_index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:

        results.append(document_chunks[idx])

    return results


if __name__ == "__main__":

    # Run ONLY once to create index
    # ingest_documents()

    # Load saved vector DB
    load_vector_store()

    # Test retrieval
    results = search_documents(
        "What is Urja?"
    )

    print("\n--- SEARCH RESULTS ---\n")

    for result in results:

        print(result)
        print("\n====================\n")