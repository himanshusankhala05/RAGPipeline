from typing import Any

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL_NAME,
)


def get_collection():
    """Create or open the local Chroma collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def add_documents(
    chunks: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
) -> None:
    collection = get_collection()

    collection.upsert(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )


def search_documents(
    question: str,
    number_of_results: int = 3,
) -> list[dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    result_count = min(number_of_results, collection.count())

    results = collection.query(
        query_texts=[question],
        n_results=result_count,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "text": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        )
    ]


def document_count() -> int:
    return get_collection().count()


def list_indexed_sources() -> list[str]:
    """Return unique source filenames currently stored in Chroma."""
    metadata_result = get_collection().get(include=["metadatas"])
    metadatas = metadata_result.get("metadatas", [])
    return list(
        dict.fromkeys(
            metadata.get("source", "Unknown")
            for metadata in metadatas
            if metadata
        )
    )


def list_indexed_document_hashes() -> set[str]:
    """Return hashes for documents indexed with duplicate protection enabled."""
    metadata_result = get_collection().get(include=["metadatas"])
    return {
        metadata["document_hash"]
        for metadata in metadata_result.get("metadatas", [])
        if metadata and metadata.get("document_hash")
    }


def delete_documents_by_source(source: str) -> int:
    """Delete all Chroma chunks belonging to one source filename."""
    collection = get_collection()
    matching_ids = collection.get(
        where={"source": source},
        include=[],
    )["ids"]

    if matching_ids:
        collection.delete(ids=matching_ids)

    return len(matching_ids)