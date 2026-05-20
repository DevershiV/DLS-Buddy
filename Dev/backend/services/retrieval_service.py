from services.ingestion_service import embedding_model, SUMMARY_INDEX_PATH, SUMMARY_CHUNKS_PATH, CONTENT_INDEX_PATH, CONTENT_CHUNKS_PATH
import faiss
import numpy as np
import pickle

def load_index(path):
    return faiss.read_index(path)

def load_chunks(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_vector_store():

    global summary_index
    global content_index

    global summary_chunks
    global content_chunks

    summary_index = load_index(
        SUMMARY_INDEX_PATH
    )

    content_index = load_index(
        CONTENT_INDEX_PATH
    )

    summary_chunks = load_chunks(
        SUMMARY_CHUNKS_PATH
    )

    content_chunks = load_chunks(
        CONTENT_CHUNKS_PATH
    )

    print("Vector stores loaded")

def search_documents(query, query_type, k=5):

    if query_type == "summary":
        index = summary_index
        document_chunks = summary_chunks
    else:
        index = content_index
        document_chunks = content_chunks

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:

        results.append(document_chunks[idx])

    return results