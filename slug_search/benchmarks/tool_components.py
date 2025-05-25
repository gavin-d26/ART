from typing import Dict, Any
import asyncio
from haystack import component
from slug_search.training.search_tools import (
    get_or_initialize_embedder_retriever_pipeline,
)


@component
class SearchDocumentsComponent:
    """
    Component that searches for relevant documents using the globally configured
    EmbedderRetrieverPipeline from search_tools.py.
    """

    def __init__(self, top_k_retriever: int = 3):
        """
        Initialize the SearchDocumentsComponent.

        Args:
            top_k_retriever: Number of top documents to retrieve
        """
        self.top_k_retriever = top_k_retriever

    @component.output_types(output_data=Dict[str, Any])
    def run(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Search for relevant documents using the query.

        Args:
            query: The search query string

        Returns:
            Dictionary containing llm_context_string and benchmark_document_chunks
        """
        return asyncio.run(self._run_async(query))

    async def _run_async(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Internal async method to search for relevant documents.

        Args:
            query: The search query string

        Returns:
            Dictionary containing llm_context_string and benchmark_document_chunks
        """
        # Get the globally configured retriever pipeline
        retriever_pipeline = get_or_initialize_embedder_retriever_pipeline(
            top_k=self.top_k_retriever
        )

        # Run the retrieval
        result = await retriever_pipeline.run_pipeline(query)

        # Extract the retrieved chunks (already in the correct format)
        retrieved_docs_for_benchmark = result.get("retrieved_chunks", [])

        # Create context string for the LLM
        llm_context_string = "\n".join(
            [doc.get("content", "") for doc in retrieved_docs_for_benchmark]
        )

        return {
            "output_data": {
                "llm_context_string": llm_context_string,
                "benchmark_document_chunks": retrieved_docs_for_benchmark,
            }
        }


@component
class ReturnFinalAnswerComponent:
    """
    Component that returns the final answer from the agent.
    """

    def __init__(self):
        """Initialize the ReturnFinalAnswerComponent."""
        pass

    @component.output_types(output_data=Dict[str, str])
    def run(self, answer: str) -> Dict[str, Dict[str, str]]:
        """
        Return the final answer.

        Args:
            answer: The final answer string

        Returns:
            Dictionary containing the final answer
        """
        return {"output_data": {"final_answer": answer}}
