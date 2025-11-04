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


class SearchQueries(BaseModel):
    """Container for one or more search queries."""

    queries: list[str]


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


def _is_positive_grade(grade: Any) -> bool:
    """Interpret model output for document relevance."""

    if isinstance(grade, DocumentGrade):
        candidate: Optional[str] = grade.binary_score
    elif isinstance(grade, dict):
        candidate = cast(Optional[str], grade.get("binary_score"))
    else:
        candidate = getattr(grade, "binary_score", None)  # type: ignore[attr-defined]

    if not isinstance(candidate, str):
        return False
    return candidate.strip().lower().startswith("y")


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, Any]:
    """Generate one or more search queries based on the conversation context."""

    configuration = Configuration.from_runnable_config(config)
    max_queries = max(1, configuration.max_queries_per_turn)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.query_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.query_model).with_structured_output(
        SearchQueries
    )

    prior_queries = "\n- ".join(state.queries) if state.queries else "<none>"
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "queries": prior_queries,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
            "max_queries": max_queries,
        },
        config,
    )
    generated = await model.ainvoke(message_value, config)

    if isinstance(generated, SearchQueries):
        candidate_queries = generated.queries
    elif isinstance(generated, dict):
        candidate_queries = cast(list[str], generated.get("queries") or [])
    else:
        candidate_queries = cast(list[str], getattr(generated, "queries", []) or [])

    cleaned_queries = [
        query.strip()
        for query in candidate_queries
        if isinstance(query, str) and query.strip()
    ][:max_queries]

    if not cleaned_queries:
        # Fall back to the user's latest message when the model fails to follow instructions.
        cleaned_queries = [get_message_text(state.messages[-1])]

    return {
        "queries": cleaned_queries,
        "current_queries": cleaned_queries,
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

    This function takes the current state and configuration, uses the most recent
    batch of queries to retrieve relevant documents using the retriever, and returns
    the aggregated results.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    queries = state.current_queries or (state.queries[-1:] if state.queries else [])

    if not queries:
        return {"retrieved_docs": []}

    retrieved: list[Document] = []
    async with retrieval.make_retriever(config) as retriever:
        for query in queries:
            docs = await retriever.ainvoke(query, config)
            for doc in docs:
                metadata = doc.metadata or {}
                metadata.setdefault("source_query", query)
                doc.metadata = metadata
            retrieved.extend(docs)

    return {"retrieved_docs": retrieved}


async def grade_documents(
    state: State, *, config: RunnableConfig
) -> dict[str, Any]:
    """Score retrieved documents for relevance and filter out low-quality results."""

    configuration = Configuration.from_runnable_config(config)
    docs = state.retrieved_docs
    question = get_message_text(state.messages[-1]) if state.messages else ""
    active_queries = state.current_queries or (
        [state.queries[-1]] if state.queries else []
    )
    current_query_text = "\n- ".join(active_queries) if active_queries else question

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
                "query": current_query_text,
                "document_number": str(index),
                "metadata": _serialize_metadata(doc.metadata),
                "document": _truncate_text(doc.page_content),
            },
            config,
        )
        grade = await grading_model.ainvoke(message_value, config)
        if _is_positive_grade(grade):
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
    max_queries = max(1, configuration.max_queries_per_turn)
    if state.refinement_attempts >= configuration.max_refinement_attempts:
        return {
            "should_retry": False,
            "discarded_docs": [],
            "retrieved_docs": [],
            "last_retrieval_relevant": None,
        }

    question = get_message_text(state.messages[-1]) if state.messages else ""
    previous_batch = state.current_queries or (
        [state.queries[-1]] if state.queries else [question]
    )
    attempt_number = state.refinement_attempts + 1
    discarded_summary = _summarize_discarded_docs(state.discarded_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.rewrite_system_prompt),
            (
                "human",
                "User question: {question}\nPrevious queries:\n{previous_queries}\nAttempt number: {attempt}\nDiscarded documents summary:\n{discarded_docs_summary}",
            ),
        ]
    )
    rewrite_model = load_chat_model(
        configuration.rewrite_model
    ).with_structured_output(SearchQueries)

    message_value = await prompt.ainvoke(
        {
            "question": question,
            "previous_queries": "\n- ".join(previous_batch),
            "attempt": attempt_number,
            "discarded_docs_summary": discarded_summary,
            "max_queries": max_queries,
        },
        config,
    )
    rewritten = await rewrite_model.ainvoke(message_value, config)

    # Structured outputs occasionally return ``None`` when the model does not adhere to
    # the schema. Fall back to the previous query in that case so the graph can continue.
    if isinstance(rewritten, SearchQueries):
        candidate_queries = rewritten.queries
    elif isinstance(rewritten, dict):
        candidate_queries = cast(list[str], rewritten.get("queries") or [])
    else:
        candidate_queries = cast(list[str], getattr(rewritten, "queries", []) or [])

    new_queries = [
        query.strip()
        for query in candidate_queries
        if isinstance(query, str) and query.strip()
    ][:max_queries]

    if not new_queries:
        new_queries = list(previous_batch)

    return {
        "queries": new_queries,
        "current_queries": new_queries,
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
