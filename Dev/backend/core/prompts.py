from langchain_core.prompts import (
    ChatPromptTemplate
)

QUERY_REWRITE_PROMPT = """
You are a query rewriting system for retrieval.

Your job is ONLY to rewrite follow-up questions
into standalone questions when necessary.

Rules:
- Preserve the original meaning.
- Do NOT ask questions.
- Do NOT answer the query.
- Do NOT add new information.
- If the query is already standalone,
  return it unchanged.

Examples:

Conversation:
User: Who killed Wenjian's father?
Follow-up:
Why did he do that?

Rewritten:
Why did the forest king kill Wenjian's father?


Conversation:
User: Give me summary of this book

Rewritten:
Give me summary of this book
"""

RAG_SYSTEM_PROMPT = """
You are a helpful AI assistant.
Keep your answers short and precise.

Use the provided context to answer.

Context:
{context}
"""

CLASSIFICATION_PROMPT = """
You are a retrieval query classifier.

Classify the query into EXACTLY ONE category:

1. summary
2. detail

Definitions:

summary:
- broad overview requests
- summaries
- themes
- high-level understanding
- overall explanations

detail:
- specific factual questions
- character actions
- events
- powers
- relationships
- precise information lookup

Examples:

Query:
What is this story about?
Category:
summary

Query:
Give me summary of this book
Category:
summary

Query:
Summarize chapter 3
Category:
summary

Query:
Who killed Wenjian's father?
Category:
detail

Query:
What is Gear Shift?
Category:
detail

Return ONLY the exact words and no other characters:
summary
OR
detail
"""

query_rewrite_prompt = (
    ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITE_PROMPT),
        ("placeholder", "{history}"),
        ("human", "{input}")
    ])
)

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