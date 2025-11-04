"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

import json
from datetime import datetime, timezone
from typing import Any, Literal, Optional, cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model

# Define the function that calls the model


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


class DocumentGrade(BaseModel):
    """Binary relevance judgement for a document."""

    binary_score: Literal["yes", "no"]


def _serialize_metadata(metadata: dict[str, Any] | None) -> str:
    """Convert document metadata into a JSON string safe for prompts."""

    if not metadata:
        return "{}"
    try:
        return json.dumps(metadata, ensure_ascii=True)
    except TypeError:
        fallback = {key: str(value) for key, value in metadata.items()}
        return json.dumps(fallback, ensure_ascii=True)


def _truncate_text(value: str, *, max_chars: int = 2000) -> str:
    """Return a length-limited view of the provided text."""

    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _summarize_discarded_docs(
    docs: list[Document], *, max_docs: int = 3, max_chars: int = 800
) -> str:
    """Produce a brief view of discarded documents for query rewriting."""

    if not docs:
        return "No discarded documents were available."

    snippets: list[str] = []
    for idx, doc in enumerate(docs[:max_docs], start=1):
        preview = _truncate_text(doc.page_content.strip(), max_chars=max_chars)
        metadata = _serialize_metadata(doc.metadata)
        snippets.append(
            f"Document {idx} metadata: {metadata}\nContent preview:\n{preview}"
        )

    remaining = len(docs) - max_docs
    if remaining > 0:
        snippets.append(f"... {remaining} additional documents omitted ...")

    return "\n\n".join(snippets)


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, Any]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, Any]: A dictionary containing the updated query list and reset tracking fields.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    messages = state.messages
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        return {
            "queries": [human_input],
            "refinement_attempts": 0,
            "last_retrieval_relevant": None,
            "should_retry": False,
            "discarded_docs": [],
            "retrieved_docs": [],
        }
    else:
        configuration = Configuration.from_runnable_config(config)
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        model = load_chat_model(configuration.query_model).with_structured_output(
            SearchQuery
        )

        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
        return {
            "queries": [generated.query],
            "refinement_attempts": 0,
            "last_retrieval_relevant": None,
            "should_retry": False,
            "discarded_docs": [],
            "retrieved_docs": [],
        }


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    async with retrieval.make_retriever(config) as retriever:
        response = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": response}


async def grade_documents(
    state: State, *, config: RunnableConfig
) -> dict[str, Any]:
    """Score retrieved documents for relevance and filter out low-quality results."""

    configuration = Configuration.from_runnable_config(config)
    docs = state.retrieved_docs
    question = get_message_text(state.messages[-1]) if state.messages else ""
    current_query = state.queries[-1] if state.queries else question

    if not docs:
        should_retry = state.refinement_attempts < configuration.max_refinement_attempts
        return {
            "retrieved_docs": [],
            "discarded_docs": [],
            "last_retrieval_relevant": False,
            "should_retry": should_retry,
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.grade_system_prompt),
            (
                "human",
                "User question: {question}\nCurrent query: {query}\nDocument number: {document_number}\nMetadata: {metadata}\nContent:\n{document}",
            ),
        ]
    )
    grading_model = load_chat_model(configuration.grade_model).with_structured_output(
        DocumentGrade
    )

    relevant_docs: list[Document] = []
    discarded_docs: list[Document] = []

    for index, doc in enumerate(docs, start=1):
        message_value = await prompt.ainvoke(
            {
                "question": question,
                "query": current_query,
                "document_number": str(index),
                "metadata": _serialize_metadata(doc.metadata),
                "document": _truncate_text(doc.page_content),
            },
            config,
        )
        grade = cast(DocumentGrade, await grading_model.ainvoke(message_value, config))
        if grade.binary_score.lower().startswith("y"):
            relevant_docs.append(doc)
        else:
            discarded_docs.append(doc)

    has_relevant = bool(relevant_docs)
    should_retry = (not has_relevant) and (
        state.refinement_attempts < configuration.max_refinement_attempts
    )

    updates: dict[str, Any] = {
        "retrieved_docs": relevant_docs,
        "discarded_docs": discarded_docs,
        "last_retrieval_relevant": has_relevant,
        "should_retry": should_retry,
    }
    if has_relevant:
        updates["refinement_attempts"] = 0

    return updates


async def rewrite_query(
    state: State, *, config: RunnableConfig
) -> dict[str, Any]:
    """Rewrite the last search query when retrieved documents are irrelevant."""

    configuration = Configuration.from_runnable_config(config)
    if state.refinement_attempts >= configuration.max_refinement_attempts:
        return {
            "should_retry": False,
            "discarded_docs": [],
            "retrieved_docs": [],
            "last_retrieval_relevant": None,
        }

    question = get_message_text(state.messages[-1]) if state.messages else ""
    previous_query = state.queries[-1] if state.queries else question
    attempt_number = state.refinement_attempts + 1
    discarded_summary = _summarize_discarded_docs(state.discarded_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.rewrite_system_prompt),
            (
                "human",
                "User question: {question}\nPrevious query: {previous_query}\nAttempt number: {attempt}\nDiscarded documents summary:\n{discarded_docs_summary}",
            ),
        ]
    )
    rewrite_model = load_chat_model(
        configuration.rewrite_model
    ).with_structured_output(SearchQuery)

    message_value = await prompt.ainvoke(
        {
            "question": question,
            "previous_query": previous_query,
            "attempt": attempt_number,
            "discarded_docs_summary": discarded_summary,
        },
        config,
    )
    rewritten = await rewrite_model.ainvoke(message_value, config)

    # Structured outputs occasionally return ``None`` when the model does not adhere to
    # the schema. Fall back to the previous query in that case so the graph can continue.
    if isinstance(rewritten, SearchQuery):
        candidate = rewritten.query
    elif isinstance(rewritten, dict):
        candidate = cast(Optional[str], rewritten.get("query"))
    else:
        candidate = getattr(rewritten, "query", None)  # type: ignore[attr-defined]

    new_query = (candidate or previous_query).strip() or previous_query

    return {
        "queries": [new_query],
        "refinement_attempts": attempt_number,
        "should_retry": False,
        "last_retrieval_relevant": None,
        "discarded_docs": [],
        "retrieved_docs": [],
    }


async def respond(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.response_model)

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph (It's just a pipe)


def route_after_grading(state: State) -> str:
    """Determine the next node after grading."""

    return "retry" if state.should_retry else "respond"


builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

builder.add_node(generate_query)  # type: ignore[arg-type]
builder.add_node(retrieve)  # type: ignore[arg-type]
builder.add_node(grade_documents)  # type: ignore[arg-type]
builder.add_node(rewrite_query)  # type: ignore[arg-type]
builder.add_node(respond)  # type: ignore[arg-type]
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "grade_documents")
builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "retry": "rewrite_query",
        "respond": "respond",
    },
)
builder.add_edge("rewrite_query", "retrieve")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "UpdatedRetrievalGraph"
