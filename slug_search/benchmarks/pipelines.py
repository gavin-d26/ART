from haystack import AsyncPipeline
from milvus_haystack.document_store import MilvusDocumentStore

# from haystack.components.embedders import SentenceTransformersTextEmbedder # Replaced
from haystack.components.embedders import (
    OpenAITextEmbedder,
)  # New for query embedding via API
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage


def build_embedded_rag_pipeline(  # Renamed function for clarity
    milvus_path: str,
    # milvus_embedding_dim: int, # Removed this parameter
    # Query Embedder (VLLM) parameters
    query_embedding_model_name: str,  # e.g., "BAAI/bge-large-en-v1.5" as served by VLLM
    query_embedder_api_base: str,  # URL for VLLM embedding server
    query_embedder_api_key: str,
    # Generator (VLLM) parameters
    generator_model_name: str,  # e.g., "gpt-3.5-turbo" or your VLLM generator model
    generator_api_base: str,  # URL for VLLM generator server
    generator_api_key: str,
    top_k_retriever: int = 3,
) -> AsyncPipeline:
    """
    Builds a RAG pipeline using OpenAITextEmbedder for queries and OpenAIChatGenerator for generation.
    Query Text -> OpenAITextEmbedder -> MilvusEmbeddingRetriever -> PromptBuilder -> OpenAIChatGenerator
    """

    document_store = MilvusDocumentStore(
        connection_args={"uri": milvus_path},
        # embedding_dim=milvus_embedding_dim, # Removed this argument
    )

    # Query Text Embedder (points to a VLLM server)
    query_text_embedder = OpenAITextEmbedder(
        api_base_url=query_embedder_api_base,
        api_key=query_embedder_api_key,
        model=query_embedding_model_name,
        # You might need to specify other parameters like dimensions if the API requires them,
        # or prefix/suffix depending on the model used.
        # For BAAI/bge models, sometimes a prefix like "query: " is used.
        # Check VLLM's OpenAI API compatibility details for embeddings.
    )

    embedding_retriever = MilvusEmbeddingRetriever(
        document_store=document_store, top_k=top_k_retriever
    )

    template = """
    Given the following documents, please answer the question.
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Question: {{query}}
    Answer:
    """
    prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(template)])

    # LLM Generator (points to another VLLM server)
    llm_chat_generator = OpenAIChatGenerator(
        model=generator_model_name,
        api_base_url=generator_api_base,
        api_key=generator_api_key,
    )

    # --- Build the AsyncPipeline ---
    rag_pipeline = AsyncPipeline()
    # Naming components clearly
    rag_pipeline.add_component("query_text_embedder", query_text_embedder)
    rag_pipeline.add_component("embedding_retriever", embedding_retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm_chat_generator", llm_chat_generator)

    # --- Connect components ---
    # 1. Query text (from pipeline input) -> query_text_embedder (expects "text")
    rag_pipeline.connect(
        "query_text_embedder.embedding", "embedding_retriever.query_embedding"
    )

    # 2. Retrieved documents -> prompt_builder (expects "documents")
    rag_pipeline.connect("embedding_retriever.documents", "prompt_builder.documents")
    # The original query text (from pipeline input) also goes to prompt_builder (expects "query")
    # This is handled by how rag_pipeline.run() is called in benchmarking.py:
    # data={"query_text_embedder": {"text": query_text}, "prompt_builder": {"query": query_text}}

    # 3. Formatted prompt -> llm_generator (expects "prompt")
    rag_pipeline.connect("prompt_builder.prompt", "llm_chat_generator.messages")

    return rag_pipeline


def build_naive_generation_pipeline(
    generator_model_name: str,
    generator_api_base: str,
    generator_api_key: str,
) -> AsyncPipeline:
    """
    Builds a Naive Generation pipeline that sends the query directly to a generator.
    Query -> PromptBuilder -> OpenAIChatGenerator
    """
    template = [ChatMessage.from_user("{{query}}")]  # Simple template to pass the query
    prompt_builder = ChatPromptBuilder(template=template)

    llm_chat_generator = OpenAIChatGenerator(
        model=generator_model_name,
        api_base_url=generator_api_base,
        api_key=generator_api_key,
    )

    naive_pipeline = AsyncPipeline()
    naive_pipeline.add_component("prompt_builder", prompt_builder)
    naive_pipeline.add_component("llm_chat_generator", llm_chat_generator)

    # Connect components:
    # The original query text (from pipeline input) goes to prompt_builder (expects "query")
    # This will be handled by how naive_pipeline.run() is called, e.g.:
    # data={"prompt_builder": {"query": query_text}}
    naive_pipeline.connect("prompt_builder.prompt", "llm_chat_generator.messages")

    return naive_pipeline
