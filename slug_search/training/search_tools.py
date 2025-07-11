from typing import List, Dict, Any, Optional

from haystack import AsyncPipeline
from milvus_haystack.document_store import MilvusDocumentStore
from milvus_haystack import MilvusEmbeddingRetriever
from haystack import (
    Document,
)  # Added for type hinting if needed, though retriever returns Document objects
from slug_search.benchmarks.pipelines import EmbedderRetrieverPipeline

# Note: Ensure 'milvus-haystack' and 'haystack-ai' (or compatible Haystack version)
# are installed in your environment.

# --- Global Document Store Configuration ---
_document_store: Optional[MilvusDocumentStore] = None
_default_milvus_db_path: str = (
    "slug_search/data/milvus_hotpotqa_better_chunking_training.db"  # Default path
)
_embedder_retriever_pipeline = None
_default_embedding_model_name = "BAAI/bge-large-en-v1.5"
_default_embedder_api_base = "http://localhost:40002/v1"
_default_embedder_api_key_env_var = "EMBEDDER_API_KEY"
_default_top_k = 3


def get_or_initialize_document_store() -> MilvusDocumentStore:
    global _default_milvus_db_path
    global _document_store
    target_path = _default_milvus_db_path
    if (
        _document_store is None
        or _document_store.connection_args.get("uri") != target_path
    ):
        _document_store = MilvusDocumentStore(
            connection_args={"uri": target_path},
        )
    return _document_store


def get_or_initialize_embedder_retriever_pipeline(
    top_k: int = 3,
) -> EmbedderRetrieverPipeline:
    global _embedder_retriever_pipeline
    if (
        _embedder_retriever_pipeline is None
        or _embedder_retriever_pipeline.pipeline is None
        or getattr(_embedder_retriever_pipeline, "top_k_retriever", None) != top_k
    ):
        _embedder_retriever_pipeline = EmbedderRetrieverPipeline(
            milvus_path=_default_milvus_db_path,
            embedding_model_name=_default_embedding_model_name,
            embedder_api_base=_default_embedder_api_base,
            embedder_api_key_env_var=_default_embedder_api_key_env_var,
            top_k_retriever=top_k,
        )
        _embedder_retriever_pipeline.top_k_retriever = top_k
    return _embedder_retriever_pipeline


async def _search_relevant_documents(search_query: str, top_k: int) -> List[str]:
    """
    Internal helper to retrieve relevant document contents.
    Args:
        search_query (str): The search query.
        top_k (int): The number of top documents to retrieve.
    Returns:
        List[str]: A list of document contents.
    """
    try:
        pipeline = get_or_initialize_embedder_retriever_pipeline(top_k=top_k)
        result = await pipeline.run_pipeline(search_query)
        result_str = "\n".join(
            [
                f"{i+1}. {doc.content}"
                for i, doc in enumerate(result.get("documents", []))
            ]
        )
        result_str = f"<search_query>\n{search_query}\n</search_query>\n\n<search_results>\n{result_str}\n</search_results>"
        return result_str
    except Exception as e:
        # Consider logging the exception here if more robust error handling is needed
        return f"Error: {e}"


async def search_documents(search_query: str) -> List[str]:
    """
    Retrieves relevant document contents from a Vector Database using vector similarity search. The search tool may sometimes return information unrelated to the search query. Do not believe the search results unless the answer to the query is directly stated in the search results, or can be reasonably inferred.

    Args:
        search_query: (str) The search query to be used to retrieve relevant documents.

    Returns:
        (str) A list of document contents as strings in a stringified format.
    """
    return await _search_relevant_documents(search_query, top_k=_default_top_k)


async def return_final_answer(answer: str) -> str:
    """
    This function is used to return the final answer to the user's query.
    It should be called with the answer passed as an argument. If you cannot find the answer, you should return "I don't know".

    Args:
        answer: (str) The answer to the user's query. If you cannot find the answer, you should return "I don't know".

    Returns:
       (str) The final answer to the user's query.
    """
    ...


def configure_search_tools(
    milvus_db_path: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    embedder_api_base: Optional[str] = None,
    embedder_api_key_env_var: Optional[str] = None,
    top_k: Optional[int] = None,
):
    global _default_milvus_db_path, _default_embedding_model_name, _default_embedder_api_base, _default_embedder_api_key_env_var, _default_top_k
    if milvus_db_path:
        _default_milvus_db_path = milvus_db_path
    if embedding_model_name:
        _default_embedding_model_name = embedding_model_name
    if embedder_api_base:
        _default_embedder_api_base = embedder_api_base
    if embedder_api_key_env_var:
        _default_embedder_api_key_env_var = embedder_api_key_env_var
    if top_k:
        _default_top_k = top_k


# Global variable to store the embedding model name
