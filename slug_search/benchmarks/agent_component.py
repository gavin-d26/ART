import json
import asyncio
from typing import Dict, Any, Optional, List
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from langchain_core.utils.function_calling import convert_to_openai_tool
from slug_search.training.search_tools import (
    search_documents,
    return_final_answer,
    get_or_initialize_embedder_retriever_pipeline,
)


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
        prompt_template: str,
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
            prompt_template: System prompt template
            search_tool_top_k: Top-k for search tool
            llm_sampling_params: Optional sampling parameters for LLM
            max_agent_steps: Maximum number of agent steps
            timeout: Timeout for LLM calls
        """
        self.prompt_template = prompt_template
        self.max_agent_steps = max_agent_steps
        self.search_tool_top_k = search_tool_top_k

        # Create tools in OpenAI format
        openai_search_documents = convert_to_openai_tool(search_documents)
        openai_return_final_answer = convert_to_openai_tool(return_final_answer)
        self.tools = [openai_search_documents, openai_return_final_answer]

        # Create LLM chat generator with tools
        generation_kwargs = llm_sampling_params or {}
        self.llm_chat_generator = OpenAIChatGenerator(
            model=llm_model_name,
            api_base_url=llm_api_base_url,
            api_key=Secret.from_env_var(llm_api_key_env_var),
            tools=self.tools,
            timeout=timeout,
            generation_kwargs=generation_kwargs,
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

        # Initialize conversation with system prompt and user query
        messages = []
        if self.prompt_template:
            messages.append(ChatMessage.from_system(self.prompt_template))
        messages.append(ChatMessage.from_user(query))

        for step in range(self.max_agent_steps):
            # Call LLM
            try:
                llm_result = self.llm_chat_generator.run(messages=messages)
            except Exception as e:
                break

            # Track generation tokens
            reply = llm_result["replies"][0]
            if reply.meta and "usage" in reply.meta:
                total_generation_tokens += reply.meta["usage"].get(
                    "completion_tokens", 0
                )

            # Check if LLM made a tool call
            tool_calls = getattr(reply, "tool_calls", None)
            if not tool_calls:
                # No tool call - append error and break
                messages.append(reply)
                messages.append(
                    ChatMessage.from_tool(
                        content="Error parsing tool call",
                        tool_call_id="no_tool_call",
                    )
                )
                final_generation = "##no_tool_called_by_llm"
                break

            # Process first tool call (matching rollout.py behavior)
            tool_call = tool_calls[0]
            tool_name = tool_call.tool_name
            tool_args_str = tool_call.arguments
            tool_id = getattr(tool_call, "id", tool_call.tool_name)
            num_tool_calls += 1

            # Early check for empty arguments
            if not tool_args_str or not tool_args_str.strip():
                messages.append(reply)
                messages.append(
                    ChatMessage.from_tool(
                        content="Error parsing tool call arguments",
                        tool_call_id=tool_id,
                    )
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

                # Add tool result to conversation
                messages.append(reply)
                messages.append(
                    ChatMessage.from_tool(
                        content=tool_result,
                        tool_call_id=tool_id,
                    )
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
                # Don't append return_final_answer tool call to messages
                break

            else:
                # Unknown tool
                tool_result = f"Unknown tool called: {tool_name}"
                messages.append(reply)
                messages.append(
                    ChatMessage.from_tool(
                        content=tool_result,
                        tool_call_id=tool_id,
                    )
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
