from typing import TypedDict, NotRequired


class GraphState(TypedDict):

    user_input: str
    history: list
    rewritten_query: NotRequired[str]
    query_type: NotRequired[str]
    retrieved_chunks: NotRequired[list]
    context: NotRequired[str]
    response: NotRequired[str]