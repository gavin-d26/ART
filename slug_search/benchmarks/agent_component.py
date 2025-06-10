import json
import asyncio
import os
from typing import Dict, Any, Optional, List
from haystack import component
from haystack.utils import Secret

from langchain_core.utils.function_calling import convert_to_openai_tool
from slug_search.training.search_tools import (
    search_documents,
    return_final_answer,
    get_or_initialize_embedder_retriever_pipeline,
)

from openai import AsyncOpenAI


@component
class CustomAgentComponent:
    """
    Custom agent component that implements the tool-calling agent loop logic,
    matching the behavior in rollout.py.
    """

    def __init__(
        self,
        llm_model_name: str,
        llm_api_base_url: str,
        llm_api_key_env_var: str,
        system_prompt: str,
        query_template: str,
        search_tool_top_k: int = 3,
        llm_sampling_params: Optional[Dict] = None,
        max_agent_steps: int = 5,
        timeout: float = 60.0 * 10,
        **kwargs,
    ):
        """
        Initialize the CustomAgentComponent.

        Args:
            llm_model_name: Name of the LLM model
            llm_api_base_url: Base URL for the LLM API
            llm_api_key_env_var: Environment variable name for API key
            system_prompt: System prompt template
            query_template: Query template with {{query}} placeholder
            search_tool_top_k: Top-k for search tool
            llm_sampling_params: Optional sampling parameters for LLM
            max_agent_steps: Maximum number of agent steps
            timeout: Timeout for LLM calls
        """
        self.llm_model_name = llm_model_name
        self.system_prompt = system_prompt
        self.query_template = query_template
        self.max_agent_steps = max_agent_steps
        self.search_tool_top_k = search_tool_top_k
        self.timeout = timeout
        self.llm_sampling_params = llm_sampling_params or {}

        # Create tools in OpenAI format (for tool definitions)
        openai_search_documents = convert_to_openai_tool(search_documents)
        openai_return_final_answer = convert_to_openai_tool(return_final_answer)
        self.tools = [openai_search_documents, openai_return_final_answer]

        # Create OpenAI client for direct API calls (matching rollout.py)
        api_key = os.getenv(llm_api_key_env_var)
        if not api_key:
            api_key = "EMPTY"  # Default for local VLLM servers

        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=llm_api_base_url,
            timeout=timeout,
        )

    @component.output_types(output_data=Dict[str, Any])
    async def run(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Run the agent loop, matching rollout.py logic.

        Args:
            query: The user query

        Returns:
            Dictionary containing generation, generation_tokens, retrieved_chunks, and tool_log
        """
        # Initialize tracking variables
        total_generation_tokens = 0
        all_retrieved_chunks = []
        tool_log = []
        final_generation = None
        num_tool_calls = 0

        # Initialize conversation with system prompt and formatted user query (matching rollout.py)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Format the query using the template (matching rollout.py behavior)
        formatted_query = self.query_template.replace("{{query}}", query)
        messages.append({"role": "user", "content": formatted_query})

        for step in range(self.max_agent_steps):
            # Call OpenAI API directly (matching rollout.py)
            try:
                # Extract sampling parameters
                temperature = self.llm_sampling_params.get("temperature", 1.0)
                top_p = self.llm_sampling_params.get("top_p", 1.0)
                max_tokens = self.llm_sampling_params.get("max_tokens", None)

                llm_response = await self.openai_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    tools=self.tools,
                    tool_choice="auto",
                    timeout=self.timeout,
                )
            except Exception as e:
                break

            # Track generation tokens
            if llm_response.usage:
                total_generation_tokens += llm_response.usage.completion_tokens

            choice = llm_response.choices[0] if llm_response.choices else None
            if not choice:
                break

            # Add assistant message to conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": (
                        [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ]
                        if choice.message.tool_calls
                        else None
                    ),
                }
            )

            tool_calls = choice.message.tool_calls
            if not tool_calls:
                # No tool call - append error and break (matching rollout.py)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": "no_tool_call",
                        "name": "no_tool_call",
                        "content": "Error parsing tool call",
                    }
                )
                final_generation = "##no_tool_called_by_llm"
                break

            # Process first tool call (matching rollout.py behavior)
            tool_call = tool_calls[0]
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            tool_id = tool_call.id
            num_tool_calls += 1

            # Early check for empty arguments
            if not tool_args_str or not tool_args_str.strip():
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": "Error parsing tool call arguments",
                    }
                )
                continue

            # Log the tool call
            try:
                tool_args = json.loads(tool_args_str)
                tool_log.append({"tool_name": tool_name, "tool_input": tool_args})
            except Exception:
                tool_log.append({"tool_name": tool_name, "tool_input": tool_args_str})

            if tool_name == "search_documents":
                try:
                    tool_args = json.loads(tool_args_str)
                    search_query = tool_args.get("search_query")

                    # Get the retriever pipeline and run it to get rich chunk data
                    retriever_pipeline = get_or_initialize_embedder_retriever_pipeline(
                        top_k=self.search_tool_top_k
                    )
                    result = await retriever_pipeline.run_pipeline(search_query)

                    # Extract retrieved chunks for benchmark output
                    retrieved_chunks = result.get("retrieved_chunks", [])
                    all_retrieved_chunks.extend(retrieved_chunks)

                    # Create context string for LLM (matching rollout.py format)
                    context_parts = []
                    for i, chunk in enumerate(retrieved_chunks):
                        content = chunk.get("content", "")
                        context_parts.append(f"{i+1}. {content}")

                    context_string = "\n".join(context_parts)
                    tool_result = f"<search_query>\n{search_query}\n</search_query>\n\n<search_results>\n{context_string}\n</search_results>"

                except Exception as e:
                    tool_result = f"Error calling tool {tool_name}: {e}"

                # Add tool result to conversation (matching rollout.py)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result,
                    }
                )
                continue

            elif tool_name == "return_final_answer":
                try:
                    tool_args = json.loads(tool_args_str)
                    answer = tool_args.get("answer")
                    if answer is not None:
                        final_generation = answer
                    else:
                        final_generation = "##no_answer_provided"
                except Exception:
                    final_generation = "##error_parsing_final_answer"
                # Don't append return_final_answer tool call to messages (matching rollout.py)
                break

            else:
                # Unknown tool (matching rollout.py)
                tool_result = f"Unknown tool called: {tool_name}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result,
                    }
                )
                final_generation = "##unknown_tool_called"
                break

        # If we exhausted max steps without final answer
        if final_generation is None:
            final_generation = "##max_steps_reached"

        return {
            "output_data": {
                "generation": final_generation,
                "generation_tokens": total_generation_tokens,
                "retrieved_chunks": all_retrieved_chunks,
                "tool_log": tool_log,
            }
        }
