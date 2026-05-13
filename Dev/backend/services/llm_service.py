from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from services.ingestion_service import search_documents, load_vector_store

from core.config import MODEL_NAME, GROQ_API_KEY

import logging

logger = logging.getLogger(__name__)

# Load Vector Store
load_vector_store()

# LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant. Keep your answers short and precise.

        Use the provided context to answer.

        Context:
        {context}
        """
    ),
    ("placeholder", "{history}"),
    ("human", "{input}")
])

# LangChain pipeline
chain = prompt | llm


def run_chain(history, user_input):
    logger.info("Running LangChain pipeline")

    try:
        logger.debug(f"History count: {len(history)}")
        logger.info(f"User input: {user_input}")

        # Retrieve relevant chunks
        retrieved_chunks = search_documents(user_input)

        # Build context
        context = "\n\n".join(retrieved_chunks)

        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks")

        response = chain.invoke({
            "history": history,
            "input": user_input,
            "context": context
        })

        logger.info("LLM response generated")
        logger.debug(f"LLM response: {response.content}")

        return response.content

    except Exception as e:
        logger.error(f"LangChain pipeline failed: {str(e)}")
        raise