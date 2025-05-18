from typing import List, Dict, Any, Optional

from milvus_haystack.document_store import MilvusDocumentStore
from milvus_haystack import MilvusEmbeddingRetriever
from haystack import (
    Document,
)  # Added for type hinting if needed, though retriever returns Document objects

# Note: Ensure 'milvus-haystack' and 'haystack-ai' (or compatible Haystack version)
# are installed in your environment.

# --- Global Document Store Configuration ---
_document_store: Optional[MilvusDocumentStore] = None
_default_milvus_db_path: str = "./milvus_pipeline.db"  # Default path
_default_collection_name: str = "DefaultCollection"  # Default collection name


def get_or_initialize_document_store() -> MilvusDocumentStore:
    global _default_milvus_db_path
    global _default_collection_name
    global _document_store
    target_path = _default_milvus_db_path
    target_collection = _default_collection_name
    if (
        _document_store is None
        or _document_store.connection_args.get("uri") != target_path
        or _document_store.collection_name != target_collection
    ):
        _document_store = MilvusDocumentStore(
            connection_args={"uri": target_path},
            collection_name=target_collection,
        )
    return _document_store


def search_relevant_documents_top_3(query_embedding: List[float]) -> List[str]:
    """Retrieves relevant document contents from a Milvus database using vector similarity search. Returns the top 3 documents.

    Args:
        query_embedding (List[float]): The embedding vector of the query.
                                        This embedding should match the dimensionality
                                    of the embeddings stored in the Milvus collection.

    Returns:
        List[str]: A list of document contents (strings).
                Returns an empty list if no documents are found or an error occurs during search
                (initialization errors will raise exceptions).
    """

    try:
        document_store = get_or_initialize_document_store()
        retriever = MilvusEmbeddingRetriever(document_store=document_store)
        result = retriever.run(
            query_embedding=query_embedding,
            top_k=3,
        )
        return [
            str(doc.content)
            for doc in result.get("documents", [])
            if doc.content is not None
        ]
    except Exception:
        return []


def search_relevant_documents_top_5(query_embedding: List[float]) -> List[str]:
    """Retrieves relevant document contents from a Milvus database using vector similarity search. Returns the top 3 documents.

    Args:
        query_embedding (List[float]): The embedding vector of the query.
                                        This embedding should match the dimensionality
                                    of the embeddings stored in the Milvus collection.

    Returns:
        List[str]: A list of document contents (strings).
                Returns an empty list if no documents are found or an error occurs during search
                (initialization errors will raise exceptions).
    """

    try:
        document_store = get_or_initialize_document_store()
        retriever = MilvusEmbeddingRetriever(document_store=document_store)
        result = retriever.run(
            query_embedding=query_embedding,
            top_k=5,
        )
        return [
            str(doc.content)
            for doc in result.get("documents", [])
            if doc.content is not None
        ]
    except Exception:
        return []
