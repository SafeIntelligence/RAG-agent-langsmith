"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever, HybridSearchRetriever
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever

from retrieval_graph.configuration import Configuration, IndexConfiguration

## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model)  # type: ignore
        case "google_genai":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@asynccontextmanager
async def make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> AsyncGenerator[VectorStoreRetriever, None]:
    """Configure this agent to connect to a specific elastic index."""

    def _build_retriever() -> VectorStoreRetriever:
        from langchain_elasticsearch import ElasticsearchStore

        connection_options = {}
        if configuration.retriever_provider == "elastic-local":
            connection_options.update(
                {
                    "es_user": os.environ["ELASTICSEARCH_USER"],
                    "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
                }
            )
        else:
            connection_options.update({"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]})

        vstore = ElasticsearchStore(
            **connection_options,  # type: ignore[arg-type]
            es_url=os.environ["ELASTICSEARCH_URL"],
            index_name="langchain_index",
            embedding=embedding_model,
        )

        search_kwargs = configuration.search_kwargs
        search_filter = search_kwargs.setdefault("filter", [])
        search_filter.append({"term": {"metadata.user_id": configuration.user_id}})
        return vstore.as_retriever(search_kwargs=search_kwargs)

    retriever = await asyncio.to_thread(_build_retriever)
    try:
        yield retriever
    finally:
        pass


@asynccontextmanager
async def make_pinecone_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> AsyncGenerator[VectorStoreRetriever, None]:
    """Configure this agent to connect to a specific pinecone index."""

    def _build_retriever() -> VectorStoreRetriever:
        from langchain_pinecone import PineconeVectorStore

        search_kwargs = configuration.search_kwargs
        search_filter = search_kwargs.setdefault("filter", {})
        search_filter.update({"user_id": configuration.user_id})
        vstore = PineconeVectorStore.from_existing_index(
            os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
        )
        return vstore.as_retriever(search_kwargs=search_kwargs)

    retriever = await asyncio.to_thread(_build_retriever)
    try:
        yield retriever
    finally:
        pass


@asynccontextmanager
async def make_mongodb_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> AsyncGenerator[VectorStoreRetriever, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""

    def _build_retriever() -> VectorStoreRetriever:
        # MongoDB driver performs platform inspection on import, so run it off the event loop.
        from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

        vstore = MongoDBAtlasVectorSearch.from_connection_string(
            os.environ["MONGODB_URI"],
            namespace="langgraph_retrieval_agent.default",
            embedding=embedding_model,
        )
        search_kwargs = configuration.search_kwargs
        # pre_filter = search_kwargs.setdefault("pre_filter", {})
        # pre_filter["user_id"] = {"$eq": configuration.user_id}
        # return vstore.as_retriever(search_kwargs=search_kwargs)
        
        retriever = MongoDBAtlasHybridSearchRetriever(
            vectorstore=vstore,
            search_index_name="vector_index",
            fulltext_penalty=50,
            vector_penalty=50,
            top_k=search_kwargs.get("k", 5),
        )
        
        return retriever

    retriever = await asyncio.to_thread(_build_retriever)
    try:
        yield retriever
    finally:
        pass


@asynccontextmanager
async def make_retriever(
    config: RunnableConfig,
) -> AsyncGenerator[VectorStoreRetriever, None]:
    """Create a retriever for the agent, based on the current configuration."""

    configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = await asyncio.to_thread(make_text_encoder, configuration.embedding_model)
    user_id = configuration.user_id
    if not user_id:
        raise ValueError("Please provide a valid user_id in the configuration.")

    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            async with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever
                return

        case "pinecone":
            async with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever
                return

        case "mongodb":
            async with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever
                return

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(Configuration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
