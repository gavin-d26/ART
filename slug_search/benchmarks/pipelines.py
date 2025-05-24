from abc import ABC, abstractmethod
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
from haystack.utils import Secret

import dotenv

dotenv.load_dotenv()


class Pipe(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initializes the pipeline. Subclasses should set up their
        specific Haystack AsyncPipeline and store it, typically in self.pipeline.
        """
        self.pipeline: AsyncPipeline | None = None
        pass

    @abstractmethod
    async def run_pipeline(self, query: str) -> dict:
        """
        Runs the asynchronous pipeline with the given query.
        Subclasses must implement this method to execute their specific pipeline.

        Args:
            query: The input query string for the pipeline.

        Returns:
            A dictionary containing the output of the pipeline.
        """
        pass


class EmbeddedRAGPipeline(Pipe):
    def __init__(
        self,
        milvus_path: str,
        query_embedding_model_name: str,
        query_embedder_api_base: str,
        query_embedder_api_key_env_var: str,
        generator_model_name: str,
        generator_api_base: str,
        generator_api_key_env_var: str,
        top_k_retriever: int = 3,
        timeout: float = 60.0 * 10,
        **kwargs,  # Accept additional parameters for flexibility
    ):
        super().__init__()
        document_store = MilvusDocumentStore(
            connection_args={"uri": milvus_path},
        )

        query_text_embedder = OpenAITextEmbedder(
            api_base_url=query_embedder_api_base,
            api_key=Secret.from_env_var(query_embedder_api_key_env_var),
            model=query_embedding_model_name,
            timeout=timeout,
        )

        embedding_retriever = MilvusEmbeddingRetriever(
            document_store=document_store, top_k=top_k_retriever
        )

        template = """
    Given the following documents given within the <documents> tags, please answer the question given within the <question> tags. Provide your final answer within 
    <answer> 
    Your Final Answer 
    </answer> tags.
    
    <documents>
    {% for doc in documents %}
        <document> {{ doc.content }} </document>
    {% endfor %}
    </documents>
    
    <question> {{query}} </question>
    
    <answer>
    """
        prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(template)])

        llm_chat_generator = OpenAIChatGenerator(
            model=generator_model_name,
            api_base_url=generator_api_base,
            api_key=Secret.from_env_var(generator_api_key_env_var),
            timeout=timeout,
        )

        rag_pipeline = AsyncPipeline()
        rag_pipeline.add_component("query_text_embedder", query_text_embedder)
        rag_pipeline.add_component("embedding_retriever", embedding_retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm_chat_generator", llm_chat_generator)

        rag_pipeline.connect(
            "query_text_embedder.embedding", "embedding_retriever.query_embedding"
        )
        rag_pipeline.connect(
            "embedding_retriever.documents", "prompt_builder.documents"
        )
        rag_pipeline.connect("prompt_builder.prompt", "llm_chat_generator.messages")

        self.pipeline = rag_pipeline

    async def run_pipeline(self, query: str) -> dict:
        if not self.pipeline:
            raise RuntimeError("Pipeline has not been initialized.")
        pipeline_input_data = {
            "query_text_embedder": {"text": query},
            "prompt_builder": {"query": query},
        }
        output = await self.pipeline.run_async(
            data=pipeline_input_data,
            include_outputs_from={"embedding_retriever", "llm_chat_generator"},
        )

        # Extract retrieved documents
        retrieved_docs = output["embedding_retriever"]["documents"]

        output_dict = {
            "generation": output["llm_chat_generator"]["replies"][0].text,
            "generation_tokens": output["llm_chat_generator"]["replies"][0].meta[
                "usage"
            ]["completion_tokens"],
            "retrieved_chunks": [
                {
                    "chunk_id": doc.meta.get("chunk_id"),
                    "original_doc_id": doc.meta.get("original_doc_id"),
                    "content": doc.content,
                    "score": getattr(doc, "score", None),  # retrieval confidence score
                }
                for doc in retrieved_docs
            ],
        }
        return output_dict


class NaiveGenerationPipeline(Pipe):
    def __init__(
        self,
        generator_model_name: str,
        generator_api_base: str,
        generator_api_key_env_var: str,
        timeout: float = 60.0 * 10,
        **kwargs,  # Accept additional parameters for flexibility
    ):
        super().__init__()
        template = [ChatMessage.from_user("{{query}}")]
        prompt_builder = ChatPromptBuilder(template=template)

        llm_chat_generator = OpenAIChatGenerator(
            model=generator_model_name,
            api_base_url=generator_api_base,
            api_key=Secret.from_env_var(generator_api_key_env_var),
            timeout=timeout,
        )

        naive_pipeline = AsyncPipeline()
        naive_pipeline.add_component("prompt_builder", prompt_builder)
        naive_pipeline.add_component("llm_chat_generator", llm_chat_generator)
        naive_pipeline.connect("prompt_builder.prompt", "llm_chat_generator.messages")

        self.pipeline = naive_pipeline

    async def run_pipeline(self, query: str) -> dict:
        if not self.pipeline:
            raise RuntimeError("Pipeline has not been initialized.")
        pipeline_input_data = {
            "prompt_builder": {"query": query},
        }
        output = await self.pipeline.run_async(data=pipeline_input_data)
        output_dict = {
            "generation": output["llm_chat_generator"]["replies"][0].text,
            "generation_tokens": output["llm_chat_generator"]["replies"][0].meta[
                "usage"
            ]["completion_tokens"],
            "retrieved_chunks": [],  # No retrieval in this pipeline
        }
        return output_dict


class EmbedderRetrieverPipeline(Pipe):
    def __init__(
        self,
        milvus_path: str,
        embedding_model_name: str,
        embedder_api_base: str,
        embedder_api_key_env_var: str,
        top_k_retriever: int = 3,
        timeout: float = 60.0 * 10,
        **kwargs,  # Accept additional parameters for flexibility
    ):
        super().__init__()
        document_store = MilvusDocumentStore(
            connection_args={"uri": milvus_path},
        )
        query_text_embedder = OpenAITextEmbedder(
            api_base_url=embedder_api_base,
            api_key=Secret.from_env_var(embedder_api_key_env_var),
            model=embedding_model_name,
            timeout=timeout,
        )
        embedding_retriever = MilvusEmbeddingRetriever(
            document_store=document_store, top_k=top_k_retriever
        )
        pipeline = AsyncPipeline()
        pipeline.add_component("query_text_embedder", query_text_embedder)
        pipeline.add_component("embedding_retriever", embedding_retriever)
        pipeline.connect(
            "query_text_embedder.embedding", "embedding_retriever.query_embedding"
        )
        self.pipeline = pipeline

    async def run_pipeline(self, query: str) -> dict:
        if not self.pipeline:
            raise RuntimeError("Pipeline has not been initialized.")
        pipeline_input_data = {
            "query_text_embedder": {"text": query},
        }
        output = await self.pipeline.run_async(
            data=pipeline_input_data, include_outputs_from={"embedding_retriever"}
        )

        retrieved_docs = output["embedding_retriever"]["documents"]

        return {
            "documents": retrieved_docs,  # Keep existing for backward compatibility
            "retrieved_chunks": [
                {
                    "chunk_id": doc.meta.get("chunk_id"),
                    "original_doc_id": doc.meta.get("original_doc_id"),
                    "content": doc.content,
                    "score": getattr(doc, "score", None),
                }
                for doc in retrieved_docs
            ],
        }
