from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import faiss
import numpy as np
import os
import pickle
import re
import sys

# Go one folder back from current script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import MODEL_NAME, GROQ_API_KEY
# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index
summary_index = None
content_index = None

# Store chunks
summary_chunks = []
content_chunks = []

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_FOLDER = os.path.join(BASE_DIR, "data")

SUMMARY_INDEX_PATH = os.path.join(BASE_DIR,"summary_index.bin")
CONTENT_INDEX_PATH = os.path.join(BASE_DIR,"content_index.bin")
SUMMARY_CHUNKS_PATH = os.path.join(BASE_DIR,"summary_chunks.pkl")
CONTENT_CHUNKS_PATH = os.path.join(BASE_DIR,"content_chunks.pkl")

summary_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
)

summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Summarize the following chapter
        in 5-8 concise lines.
        """
    ),
    ("human", "{text}")
])

summary_chain = summary_prompt | summary_llm

def load_documents():

    all_chapters = []

    for file in os.listdir(DATA_FOLDER):

        if file.endswith(".docx"):

            path = os.path.join(
                DATA_FOLDER,
                file
            )

            loader = Docx2txtLoader(path)

            documents = loader.load()

            for doc in documents:

                chapters = split_into_chapters(
                    doc.page_content
                )

                for idx, chapter in enumerate(chapters):

                    chapter["source"] = file

                    chapter["chapter_number"] = idx + 1

                    all_chapters.append(chapter)
    
    return all_chapters

def split_into_chapters(text):

    pattern = r"(Chapter\s+\d+)"

    parts = re.split(pattern, text)

    chapters = []

    for i in range(1, len(parts), 2):

        title = parts[i]

        content = parts[i + 1]

        chapters.append({
            "chapter_title": title,
            "content": content
        })
    # Fallback if no chapters found
    if not chapters:

        chapters.append({
            "chapter_title": "Full Document",
            "content": text
        })

    return chapters

def generate_summary(text):

    response = summary_chain.invoke({
        "text": text
    })

    return response.content

def ingest_documents():

    global summary_index
    global content_index
    global summary_chunks
    global content_chunks

    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # chunks = []

    for chapter in documents:

        # Generate chapter summary
        chapter_summary = generate_summary(
            chapter["content"]
        )

        # Store summary separately
        summary_chunks.append({
            "content": chapter_summary,
            "metadata": {
                "type": "chapter_summary",
                "source": chapter["source"],
                "chapter_title": chapter["chapter_title"],
                "chapter_number": chapter["chapter_number"]
            }
        })

        # Chunk detailed content
        split_chunks = splitter.create_documents(
            [chapter["content"]]
        )

        for chunk in split_chunks:
            content_chunks.append({
                "content": chunk.page_content,
                "metadata": {
                    "type": "content_chunk",
                    "source": chapter["source"],
                    "chapter_title": chapter["chapter_title"],
                    "chapter_number": chapter["chapter_number"]
                }
            })

    print(f"Created {len(summary_chunks)} chunks")
    print(f"Created {len(content_chunks)} chunks")

    summary_index = create_vector_store(summary_chunks)
    content_index = create_vector_store(content_chunks)

    save_index(summary_index, SUMMARY_INDEX_PATH)
    save_index(content_index, CONTENT_INDEX_PATH)

    save_chunks(summary_chunks, SUMMARY_CHUNKS_PATH)
    save_chunks(content_chunks, CONTENT_CHUNKS_PATH)

def save_index(faiss_index, path):
    faiss.write_index(
        faiss_index,
        path
    )
def save_chunks(document_chunks, path):
    # Save chunks
    with open(path, "wb") as f:

        pickle.dump(document_chunks, f)

    print("FAISS index saved successfully")

def create_vector_store(chunks):

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index

if __name__ == "__main__":

    # Run ONLY once to create index
    ingest_documents()

    # Load saved vector DB
    # load_vector_store()

    # # Test retrieval
    # results = search_documents(
    #     "What is Urja?"
    # )

    # print("\n--- SEARCH RESULTS ---\n")

    # for result in results:

    #     print(result)
    #     print("\n====================\n")