"""
ChromaDB vector store for customer payment history and dispute notes.
Used to retrieve relevant context before generating dunning emails,
so the email reflects actual relationship history, not just invoice data.
"""

import os
import chromadb
from chromadb.config import Settings

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./db/chroma")
COLLECTION_NAME = "customer_history"


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


def upsert_customer_note(customer_id: str, note: str, metadata: dict | None = None) -> None:
    col = _get_collection()
    col.upsert(
        ids=[customer_id],
        documents=[note],
        metadatas=[metadata or {}],
    )


def retrieve_customer_context(customer_id: str, n_results: int = 3) -> str:
    col = _get_collection()
    try:
        results = col.query(query_texts=[customer_id], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    except Exception:
        return ""
