from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from services.retrieval_service import search_documents, load_vector_store

from core.config import MODEL_NAME, GROQ_API_KEY
from core.prompts import query_rewrite_prompt, rag_prompt, classification_prompt

import logging

logger = logging.getLogger(__name__)

# Load Vector Store
load_vector_store()

# LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
)

# LangChain pipeline
query_rewrite_chain = query_rewrite_prompt | llm
rag_chain = rag_prompt | llm
classification_chain = classification_prompt | llm

def rewrite_query(history,user_input):
    if not history:
        return user_input

    response = query_rewrite_chain.invoke({"history": history,"input": user_input})
    rewritten_query = (response.content.strip())

    return rewritten_query

def run_chain(history, user_input):
    logger.info("Running LangChain pipeline")

    try:
        logger.debug(f"History count: {len(history)}")
        logger.info(f"User input: {user_input}")

        # Rewrite user question based on History to tackle follow ups
        rewritten_query = rewrite_query(history,user_input)

        logger.info(f"Rewritten query: {rewritten_query}")

        # Classify question into either summary or specific
        response = classification_chain.invoke({"query": rewritten_query})

        query_type = (
            response.content
            .strip()
            .lower()
        )
        logger.info(f"Query classified as: {query_type}")

        # Retrieve relevant chunks
        retrieved_chunks = search_documents(rewritten_query, query_type)

        # Build context
        context = "\n\n".join(
            chunk["content"]
            for chunk in retrieved_chunks
        )
        print(context)

        for chunk in retrieved_chunks:
            logger.info(
                f"Retrieved from "
                f"{chunk['metadata']}"
            )

        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks")

        response = rag_chain.invoke({
            "history": history,
            "input": rewritten_query,
            "context": context
        })

        logger.info("LLM response generated")
        logger.debug(f"LLM response: {response.content}")

        return response.content

    except Exception as e:
        logger.error(f"LangChain pipeline failed: {str(e)}")
        raise