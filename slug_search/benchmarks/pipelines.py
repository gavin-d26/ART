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
import json
import re
from typing import Optional, Dict

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
        top_k_retriever: int = 5,
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

        # Extract the raw generation text
        raw_generation = output["llm_chat_generator"]["replies"][0].text

        # Extract answer from <answer> tags, fallback to raw text if no tags found
        extracted_answer = extract_answer_from_tags(raw_generation, "answer")

        output_dict = {
            "generation": extracted_answer,
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
        template = """
    Please answer the question given within the <question> tags. Provide your final answer within 
    <answer> 
    Your Final Answer 
    </answer> tags.
    
    <question> {{query}} </question>
    """
        prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(template)])

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

        # Extract the raw generation text
        raw_generation = output["llm_chat_generator"]["replies"][0].text

        # Extract answer from <answer> tags, fallback to raw text if no tags found
        extracted_answer = extract_answer_from_tags(raw_generation, "answer")

        output_dict = {
            "generation": extracted_answer,
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
        top_k_retriever: int = 5,
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
            # Add connection limits to prevent too many concurrent connections
            max_retries=3,
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


class AgenticToolCallingPipeline(Pipe):
    """
    Pipeline that implements an agentic tool-calling system for benchmarking.
    """

    def __init__(
        self,
        agent_llm_model_name: str,
        agent_llm_api_base_url: str,
        agent_llm_api_key_env_var: str,
        agent_query_prompt_template_key: str,
        agent_system_prompt_key: str = "qwen_2.5_3b_instruct_system_prompt",
        search_tool_top_k: int = 5,
        agent_llm_sampling_params: Optional[Dict] = None,
        max_agent_steps: int = 5,
        timeout: float = 60.0 * 10,
        **kwargs,
    ):
        """
        Initialize the AgenticToolCallingPipeline.

        Args:
            agent_llm_model_name: Name of the LLM model for the agent
            agent_llm_api_base_url: Base URL for the agent LLM API
            agent_llm_api_key_env_var: Environment variable name for API key
            agent_query_prompt_template_key: Key to load query template from prompts.json
            agent_system_prompt_key: Key to load system prompt from prompts.json
            search_tool_top_k: Top-k for search tool
            agent_llm_sampling_params: Optional sampling parameters for LLM
            max_agent_steps: Maximum number of agent steps
            timeout: Timeout for LLM calls
            **kwargs: Additional parameters for flexibility
        """
        super().__init__()

        # Import here to avoid circular imports
        from slug_search.benchmarks.agent_component import CustomAgentComponent

        # Load both system prompt and query template from prompts.json
        try:
            with open("slug_search/training/prompts.json", "r") as f:
                prompts = json.load(f)

            # Load system prompt
            system_prompt = prompts.get(agent_system_prompt_key, "")
            if not system_prompt:
                raise ValueError(
                    f"System prompt key '{agent_system_prompt_key}' not found in prompts.json"
                )

            # Load query template
            query_template = prompts.get(agent_query_prompt_template_key, "")
            if not query_template:
                raise ValueError(
                    f"Query template key '{agent_query_prompt_template_key}' not found in prompts.json"
                )
        except Exception as e:
            raise ValueError(f"Failed to load prompt templates: {str(e)}")

        # Create the custom agent component
        custom_agent_component = CustomAgentComponent(
            llm_model_name=agent_llm_model_name,
            llm_api_base_url=agent_llm_api_base_url,
            llm_api_key_env_var=agent_llm_api_key_env_var,
            system_prompt=system_prompt,
            query_template=query_template,
            search_tool_top_k=search_tool_top_k,
            llm_sampling_params=agent_llm_sampling_params,
            max_agent_steps=max_agent_steps,
            timeout=timeout,
        )

        # Create the pipeline
        self.pipeline = AsyncPipeline()
        self.pipeline.add_component("custom_agent", custom_agent_component)

    async def run_pipeline(self, query: str) -> dict:
        """
        Run the agentic tool-calling pipeline.

        Args:
            query: The input query string

        Returns:
            Dictionary containing generation, generation_tokens, retrieved_chunks, and tool_log
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline has not been initialized.")

        pipeline_input_data = {"custom_agent": {"query": query}}

        output = await self.pipeline.run_async(
            data=pipeline_input_data, include_outputs_from={"custom_agent"}
        )

        # Extract the output from the custom agent
        agent_output = output["custom_agent"]["output_data"]

        return {
            "generation": agent_output.get("generation", ""),
            "generation_tokens": agent_output.get("generation_tokens", 0),
            "retrieved_chunks": agent_output.get("retrieved_chunks", []),
            "tool_log": agent_output.get("tool_log", []),
        }


def extract_answer_from_tags(text: str, tag: str = "answer") -> str:
    """
    Extract content from XML-like tags in the text.

    Args:
        text: The input text containing tags
        tag: The tag name to extract from (default: "answer")

    Returns:
        The extracted content, or the original text if no tags found
    """
    # Pattern to match opening and closing tags with content in between
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"

    # Search for the pattern (case-insensitive, multiline, dotall)
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)

    if match:
        # Return the content inside the tags, stripped of leading/trailing whitespace
        return match.group(1).strip()

    # If no tags found, return the original text
    return text
