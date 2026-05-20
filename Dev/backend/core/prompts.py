from langchain_core.prompts import (
    ChatPromptTemplate
)

RAG_SYSTEM_PROMPT = """
You are a helpful AI assistant.
Keep your answers short and precise.

Use the provided context to answer.

Context:
{context}
"""

CLASSIFICATION_PROMPT = """
Classify the query into ONE category:

1. summary
2. detail

Rules:
- summary:
  broad overview questions

- detail:
  specific factual questions

Return ONLY the exact words and no other characters:
summary
OR
detail
"""

rag_prompt = (
    ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("placeholder", "{history}"),
        ("human", "{input}")
    ])
)

classification_prompt = (
    ChatPromptTemplate.from_messages([
        ("system", CLASSIFICATION_PROMPT),
        ("human", "{query}")
    ])
)