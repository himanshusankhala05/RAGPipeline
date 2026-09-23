import re
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


def _normalize_keyword_tokens(value: str) -> set[str]:
    """Return normalized keyword tokens for simple lexical matching."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token and token not in {"the", "and", "with", "from", "that", "this"}
    }


def _keyword_score(question: str, document_text: str) -> float:
    """Score document relevance using shared search term overlap."""
    query_tokens = _normalize_keyword_tokens(question)
    if not query_tokens:
        return 0.0

    document_tokens = _normalize_keyword_tokens(document_text)
    if not document_tokens:
        return 0.0

    overlap = query_tokens & document_tokens
    return len(overlap) / max(len(query_tokens), 1)


def _merge_search_results(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Combine semantic and lexical rankings into a single result list."""
    ranked: dict[tuple[str, str], dict[str, Any]] = {}

    for result in semantic_results:
        key = (result["metadata"].get("source", "unknown"), result["text"])
        ranked[key] = {
            **result,
            "_combined_score": 1.0 / (1.0 + max(float(result.get("distance", 0.0)), 0.0)),
        }

    for result in keyword_results:
        key = (result["metadata"].get("source", "unknown"), result["text"])
        if key in ranked:
            ranked[key]["_combined_score"] += result.get("_keyword_score", 0.0)
        else:
            ranked[key] = {
                **result,
                "_combined_score": result.get("_keyword_score", 0.0),
            }

    merged = []
    for entry in ranked.values():
        score = entry.pop("_combined_score", 0.0)
        entry["_score"] = score
        merged.append(entry)

    merged.sort(key=lambda item: float(item.get("_score", 0.0)), reverse=True)
    return merged[:limit]


def _generate_hyde_queries(question: str) -> list[str]:
    """Create a small set of plausible answer-like queries for HyDE retrieval."""
    prompt = question.strip()
    if not prompt:
        return []
    clauses = [prompt, f"What is {prompt}?", f"Explain {prompt} in detail."]
    seen: set[str] = set()
    queries: list[str] = []
    for clause in clauses:
        normalized = clause.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            queries.append(normalized)
    return queries


def get_search_queries(question: str, search_mode: str) -> list[str]:
    """Return the queries generated for the chosen retrieval strategy."""
    mode = (search_mode or "semantic").lower()
    if mode == "multi_query":
        return _generate_multi_queries(question)
    if mode == "hyde":
        return _generate_hyde_queries(question)
    return [question] if question.strip() else []


def _generate_multi_queries(question: str) -> list[str]:
    """Generate multiple paraphrased question variations for broader retrieval."""
    prompt = question.strip()
    if not prompt:
        return []
    expanded = [
        prompt,
        f"Find information about {prompt}",
        f"What does {prompt} refer to?",
        f"Explain {prompt} in context.",
    ]
    seen: set[str] = set()
    queries: list[str] = []
    for item in expanded:
        normalized = item.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            queries.append(normalized)
    return queries


def rerank_documents(
    question: str,
    documents: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Reorder retrieved chunks by semantic closeness plus keyword overlap."""
    if not documents:
        return []

    reranked_results = []
    for document in documents:
        text = str(document.get("text", ""))
        distance = document.get("distance")
        semantic_score = 1.0
        if distance is not None:
            try:
                semantic_score = 1.0 / (1.0 + max(float(distance), 0.0))
            except (TypeError, ValueError):
                semantic_score = 1.0

        keyword_score = _keyword_score(question, text)
        combined = (0.7 * semantic_score) + (0.3 * keyword_score)
        reranked_results.append({
            **document,
            "_rerank_score": combined,
        })

    reranked_results.sort(key=lambda item: float(item.get("_rerank_score", 0.0)), reverse=True)
    if limit is not None:
        reranked_results = reranked_results[:limit]
    return reranked_results


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
    where: dict[str, Any] | None = None,
    search_mode: str = "semantic",
) -> list[dict[str, Any]]:
    """Search Chroma using a selected retrieval strategy."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    result_count = min(max(number_of_results, 1), collection.count())
    search_mode = (search_mode or "semantic").lower()

    if search_mode == "keyword":
        collection_data = collection.get(where=where, include=["documents", "metadatas"])
        documents = collection_data.get("documents", [])
        metadatas = collection_data.get("metadatas", [])

        keyword_results = []
        for document, metadata in zip(documents, metadatas, strict=False):
            score = _keyword_score(question, document)
            if score <= 0:
                continue
            keyword_results.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "_keyword_score": score,
                    "distance": 1.0 - score,
                }
            )

        keyword_results.sort(key=lambda item: item["_keyword_score"], reverse=True)
        return rerank_documents(question, keyword_results[:result_count], result_count)

    if search_mode == "hybrid":
        semantic_query = collection.query(
            query_texts=[question],
            n_results=min(max(number_of_results * 3, 5), collection.count()),
            where=where,
        )
        semantic_documents = semantic_query["documents"][0]
        semantic_metadatas = semantic_query["metadatas"][0]
        semantic_distances = semantic_query["distances"][0]

        collection_data = collection.get(where=where, include=["documents", "metadatas"])
        keyword_results = []
        for document, metadata in zip(
            collection_data.get("documents", []),
            collection_data.get("metadatas", []),
            strict=False,
        ):
            score = _keyword_score(question, document)
            if score <= 0:
                continue
            keyword_results.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "_keyword_score": score,
                }
            )

        semantic_results = [
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
            for document, metadata, distance in zip(
                semantic_documents,
                semantic_metadatas,
                semantic_distances,
                strict=False,
            )
        ]
        merged_results = _merge_search_results(semantic_results, keyword_results, result_count)
        return rerank_documents(question, merged_results, result_count)

    if search_mode == "multi_query":
        candidate_queries = _generate_multi_queries(question)
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidate_queries:
            query_kwargs: dict[str, Any] = {
                "query_texts": [candidate],
                "n_results": max(2, min(4, result_count)),
            }
            if where:
                query_kwargs["where"] = where
            response = collection.query(**query_kwargs)
            for document, metadata, distance in zip(
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
                strict=False,
            ):
                key = (metadata.get("source", "unknown"), document)
                if key in seen:
                    continue
                seen.add(key)
                combined.append({
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                })
        return rerank_documents(question, combined[:result_count], result_count)

    if search_mode == "hyde":
        candidate_queries = _generate_hyde_queries(question)
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidate_queries:
            query_kwargs: dict[str, Any] = {
                "query_texts": [candidate],
                "n_results": max(2, min(4, result_count)),
            }
            if where:
                query_kwargs["where"] = where
            response = collection.query(**query_kwargs)
            for document, metadata, distance in zip(
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
                strict=False,
            ):
                key = (metadata.get("source", "unknown"), document)
                if key in seen:
                    continue
                seen.add(key)
                combined.append({
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                })
        return rerank_documents(question, combined[:result_count], result_count)

    query_kwargs: dict[str, Any] = {
        "query_texts": [question],
        "n_results": result_count,
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    semantic_results = [
        {
            "text": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
            strict=False,
        )
    ]
    return rerank_documents(question, semantic_results, result_count)


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