"""CLI utility to ingest HTML documents into MongoDB Atlas vector store.

This script prepares documents for the retrieval agent by:
- Converting HTML files to clean text chunks.
- Embedding the text with the configured encoder.
- Writing the documents to MongoDB Atlas using the same schema expected by
  :func:`retrieval_graph.retrieval.make_mongodb_retriever`.
- Creating (or validating) the Atlas Vector Search index required for
  similarity search.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import BSHTMLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from tqdm import tqdm

from parse_html import get_chunked_texts


with open("/Users/atriviveksharma/Desktop/SafeIntelligence/LLM_experiments/google_api_key.txt","r") as f:
    google_key = f.read()
    
os.environ["GOOGLE_API_KEY"] = google_key

LOGGER = logging.getLogger("retrieval_graph.scripts.mongodb_ingest")


@dataclass
class IngestionStats:
    """Tracks high-level ingestion metrics."""

    files_processed: int = 0
    chunks_created: int = 0
    bytes_read: int = 0

    def log(self) -> None:
        LOGGER.info(
            "Processed %d files → %d chunks (%.2f KB)",
            self.files_processed,
            self.chunks_created,
            self.bytes_read / 1024 if self.bytes_read else 0,
        )


def _iter_html_files(root: Path) -> Iterable[Path]:
    return sorted(p for p in root.rglob("*.html") if p.is_file())


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def build_documents(
    html_dir: Path,
    user_id: str,
    chunk_size: int,
    chunk_overlap: int,
    max_files: int | None = None,
) -> tuple[list[Document], list[str], IngestionStats]:
    """Load, clean, and chunk HTML documents into LangChain `Document`s."""

    stats = IngestionStats()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    documents: list[Document] = []
    doc_ids: list[str] = []
    
    total_files = len(list(_iter_html_files(html_dir)))
    if max_files is not None:
        total_files = min(total_files, max_files)

    counter = 0

    for html_path in tqdm(_iter_html_files(html_dir), total=total_files):
        stats.files_processed += 1
        try:
            stats.bytes_read += html_path.stat().st_size
        except OSError:
            LOGGER.debug("Could not determine size for %s", html_path)

        loader = BSHTMLLoader(str(html_path))
        try:
            loaded_docs = loader.load()
        except Exception as exc:
            LOGGER.warning("Skipping %s due to loader error: %s", html_path, exc)
            continue

        if not loaded_docs:
            LOGGER.warning("Skipping empty HTML file: %s", html_path)
            continue

        relative_source = _relative_path(html_path, html_dir)
        for doc in loaded_docs:
            doc.metadata.setdefault("source", relative_source)
            doc.metadata["abs_path"] = str(html_path)
            doc.metadata["user_id"] = user_id

        # Use BeautifulSoup-backed loader output for consistent chunking.
        chunks = splitter.split_documents(loaded_docs)
        if not chunks:
            LOGGER.warning("No text chunks created from %s", html_path)
            continue

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            documents.append(chunk)
            doc_ids.append(f"{user_id}::{relative_source}::chunk-{idx}")

        stats.chunks_created += len(chunks)
        counter += 1
        if max_files is not None and counter >= max_files:
            break
        
    return documents, doc_ids, stats



def ingest_html_folder(args: argparse.Namespace) -> None:
    html_dir = args.html_dir.expanduser().resolve()
    if not html_dir.exists() or not html_dir.is_dir():
        raise FileNotFoundError(f"HTML directory not found: {html_dir}")

    LOGGER.info("Loading HTML files from %s", html_dir)
    # documents, doc_ids, stats = build_documents(
    #     html_dir=html_dir,
    #     user_id=args.user_id,
    #     chunk_size=args.chunk_size,
    #     chunk_overlap=args.chunk_overlap,
    #     max_files=args.max_files,
    # )
    
    documents: list[Document] = []
    doc_ids: list[str] = []
    
    counter = 0

    max_files = args.max_files

    total_files = len(list(_iter_html_files(html_dir)))
    if max_files is not None:
        total_files = min(total_files, max_files)

    for html_fp in tqdm(_iter_html_files(html_dir), total=total_files):
        # html_fp = os.path.join(html_dir, fp)
        docs, ids = get_chunked_texts(str(html_fp), args.user_id)
        documents.extend(docs)
        doc_ids.extend(ids)
        
        if args.max_files is not None and counter >= args.max_files:
            break
        
        counter += 1

    if not documents:
        LOGGER.warning("No documents prepared for ingestion. Exiting.")
        return

    # stats.log()

    if args.dry_run:
        LOGGER.info("Dry-run enabled; skipping MongoDB writes.")
        return

    uri = args.mongodb_uri or os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "Missing MongoDB connection string. Supply via --mongodb-uri or MONGODB_URI env var."
        )

    LOGGER.info("Initialising embedding model: %s", args.embedding_model)
    
    

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_document")
    
    # embedding_model = make_text_encoder(args.embedding_model)
    
    
    
    probe_vector = embedding_model.embed_query("dimension probe")
    embedding_dim = len(probe_vector)
    LOGGER.debug("Embedding dimension detected: %d", embedding_dim)

    namespace = f"{args.database}.{args.collection}"
    LOGGER.info("Connecting to MongoDB namespace: %s", namespace)

    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        uri,
        namespace=namespace,
        embedding=embedding_model,
        index_name=args.index_name,
    )

    batch_size = max(args.batch_size, 1)
    total_docs = len(documents)
    LOGGER.info("Adding %d documents to MongoDB (batch size: %d)", total_docs, batch_size)
    with tqdm(total=total_docs, desc="Uploading", unit="doc") as progress:
        for start in range(0, total_docs, batch_size):
            end = min(start + batch_size, total_docs)
            vector_store.add_documents(documents[start:end], ids=doc_ids[start:end])
            progress.update(end - start)
    
    
    LOGGER.info("Creating vector search index '%s'", args.index_name)
    # Index user_id so Atlas can honour per-user filters downstream.
    vector_store.create_vector_search_index(
        dimensions=embedding_dim,
        filter_fields=["metadata.user_id", "user_id"],
    )

    LOGGER.info("Ingestion complete.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a directory of HTML documents into MongoDB Atlas with vector search support."
        )
    )
    parser.add_argument(
        "--html_dir",
        type=Path,
        help="Path to the folder containing HTML files (searched recursively).",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Value stored in metadata.user_id for all ingested documents.",
    )
    parser.add_argument(
        "--database",
        default="langgraph_retrieval_agent",
        help="MongoDB database name (default: langgraph_retrieval_agent).",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="MongoDB collection name (default: default).",
    )
    parser.add_argument(
        "--index-name",
        default="vector_index",
        help="Atlas vector search index name (default: vector_index).",
    )
    parser.add_argument(
        "--embedding-model",
        default="openai/text-embedding-3-small",
        help="Embedding model identifier (provider/model).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Target character count per chunk (default: 5000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Character overlap between consecutive chunks (default: 200).",
    )
    parser.add_argument(
        "--mongodb-uri",
        dest="mongodb_uri",
        help="MongoDB connection string. Defaults to the MONGODB_URI environment variable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk documents without writing to MongoDB.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of HTML files to process (default: all files).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of documents to upload per batch (default: 64).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        ingest_html_folder(args)
    except Exception:  # pragma: no cover - surface errors for CLI users
        LOGGER.exception("Ingestion failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
