from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from core.config import MODEL_NAME, GROQ_API_KEY
from core.prompts import query_rewrite_prompt, rag_prompt, classification_prompt
from services.retrieval_service import search_documents, load_vector_store
from langchain_groq import ChatGroq
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

# Defining the state Class
class GraphState(TypedDict):

    user_input: str
    history: list
    rewritten_query: NotRequired[str]
    query_type: NotRequired[str]
    retrieved_chunks: NotRequired[list]
    context: NotRequired[str]
    response: NotRequired[str]

def rewrite_node(state: GraphState):
    logger.info("Rewrite Node")

    if not state["history"]:
        return {"rewritten_query": state["user_input"]}
    response = query_rewrite_chain.invoke({"history": state["history"], "input": state["user_input"]})
    rewritten_query = response.content.strip()

    logger.info(f"Rewritten query: {rewritten_query}")
    return {
        "rewritten_query": rewritten_query
    }

def classify_node(state: GraphState):
    logger.info("Classify Node")

    response = classification_chain.invoke({"query": state["rewritten_query"]})
    query_type = (response.content.strip().lower())

    logger.info(f"Query classified as: {query_type}")
    return {
        "query_type": query_type
    }

def retrieve_document_node(state: GraphState):
    logger.info("Document Retrieval")

    chunks = search_documents(
        state["rewritten_query"],
        state["query_type"]
    )

    logger.info(f"Retrieved {len(chunks)} chunks")
    for chunk in chunks:
        logger.info(f"Retrieved from {chunk['metadata']}")
    return {
        "retrieved_chunks": chunks
    }

def context_node(state: GraphState):
    logger.info("Context Node")
    context = "\n\n".join(
        chunk["content"]
        for chunk in state["retrieved_chunks"]
    )
    return {
        "context": context
    }

def response_node(state: GraphState):
    logger.info("Response Node")
    response = rag_chain.invoke({
        "history": state["history"],
        "input": state["user_input"],
        "context": state["context"]
    })

    logger.info(f"Generated response: {response.content}")
    return {
        "response": response.content
    }

graph_builder = StateGraph(GraphState)

graph_builder.add_node("rewrite_node",rewrite_node)
graph_builder.add_node("classify_node",classify_node)
graph_builder.add_node("retrieve_document",retrieve_document_node)
graph_builder.add_node("context_node",context_node)
graph_builder.add_node("response_node",response_node)

graph_builder.set_entry_point("rewrite_node")

graph_builder.add_edge("rewrite_node","classify_node")
graph_builder.add_edge("classify_node","retrieve_document")
graph_builder.add_edge("retrieve_document","context_node")
graph_builder.add_edge("context_node","response_node")
graph_builder.add_edge("response_node",END)

graph = graph_builder.compile()
# print(graph.get_graph().draw_ascii())

# result = graph.invoke({
#     "user_input": "Why did he do that?",
#     "history": [
#         {
#             "role": "user",
#             "content": "Who killed Wenjian's father?"
#         }
#     ]
# })

# print(result)