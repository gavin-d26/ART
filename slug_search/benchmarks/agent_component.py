import json
import asyncio
from typing import Dict, Any, Optional, List
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.tools import ComponentTool
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from slug_search.benchmarks.tool_components import (
    SearchDocumentsComponent,
    ReturnFinalAnswerComponent,
)


@component
class CustomAgentComponent:
    """
    Custom agent component that implements the tool-calling agent loop logic.
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
        unknown_tool_handling_strategy: str = "break_loop",
        timeout: float = 60.0 * 10,
    ):
        """
        Initialize the CustomAgentComponent.

        Args:
            llm_model_name: Name of the LLM model
            llm_api_base_url: Base URL for the LLM API
            llm_api_key_env_var: Environment variable name for API key
            prompt_template: Template for formatting the user query (should contain {{query}} placeholder)
            search_tool_top_k: Top-k for search tool
            llm_sampling_params: Optional sampling parameters for LLM
            max_agent_steps: Maximum number of agent steps
            unknown_tool_handling_strategy: Strategy for handling unknown tools ("error_to_model" or "break_loop")
            timeout: Timeout for LLM calls
        """
        self.prompt_template = prompt_template
        self.max_agent_steps = max_agent_steps
        self.unknown_tool_handling_strategy = unknown_tool_handling_strategy

        # Create tool components
        search_documents_component = SearchDocumentsComponent(
            top_k_retriever=search_tool_top_k
        )
        return_final_answer_component = ReturnFinalAnswerComponent()

        # Create ComponentTool wrappers
        search_tool = ComponentTool(
            component=search_documents_component,
            name="search_documents",
            description="Search for relevant documents using a query. Returns context information.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents",
                }
            },
        )

        return_tool = ComponentTool(
            component=return_final_answer_component,
            name="return_final_answer",
            description="Return the final answer to the user's query.",
            parameters={
                "answer": {
                    "type": "string",
                    "description": "The final answer to return to the user",
                }
            },
        )

        self.tools = [search_tool, return_tool]
        self.tool_invoker = ToolInvoker(tools=self.tools)

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
    def run(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Run the agent loop.

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

        # Initialize conversation with formatted user query (no system message)
        # Apply the prompt template to the user query
        formatted_query = self.prompt_template.replace("{{query}}", query)
        messages = [
            ChatMessage.from_user(formatted_query),
        ]

        for step in range(self.max_agent_steps):
            # Call LLM
            llm_result = self.llm_chat_generator.run(messages=messages)

            # Track generation tokens
            reply = llm_result["replies"][0]
            if reply.meta and "usage" in reply.meta:
                total_generation_tokens += reply.meta["usage"].get(
                    "completion_tokens", 0
                )

            # Check if LLM made a tool call
            if not reply.tool_calls:
                # No tool call - break loop
                final_generation = "##no_tool_called_by_llm"
                break

            # Process tool calls
            tool_call_successful = False
            for tool_call in reply.tool_calls:
                tool_name = tool_call.tool_name
                tool_args = tool_call.arguments

                # Log the tool call
                tool_log.append({"tool_name": tool_name, "tool_input": tool_args})

                # Check if tool exists
                if tool_name not in [tool.name for tool in self.tools]:
                    # Unknown tool
                    if self.unknown_tool_handling_strategy == "break_loop":
                        final_generation = "##unknown_tool_called"
                        tool_call_successful = False
                        break
                    elif self.unknown_tool_handling_strategy == "error_to_model":
                        # Add error message to conversation
                        error_msg = f"Error: Unknown tool '{tool_name}' called."
                        messages.append(reply)
                        messages.append(
                            ChatMessage.from_tool(
                                content=error_msg, tool_call_id=tool_call.id
                            )
                        )
                        continue

                # Execute tool
                try:
                    # Create a ChatMessage with the tool call for the ToolInvoker
                    tool_call_message = ChatMessage.from_assistant(
                        tool_calls=[tool_call]
                    )
                    tool_result = self.tool_invoker.run(messages=[tool_call_message])

                    # Extract the tool result from the response
                    tool_messages = tool_result.get("tool_messages", [])
                    if tool_messages:
                        tool_message = tool_messages[0]
                        tool_call_result = tool_message.tool_call_result

                        if tool_call_result and not tool_call_result.error:
                            # Handle search_documents tool result
                            if tool_name == "search_documents":
                                # Parse the result string back to dict if needed
                                result_str = tool_call_result.result
                                try:
                                    import json
                                    import ast

                                    # First try JSON parsing
                                    try:
                                        output_data = json.loads(result_str)
                                    except json.JSONDecodeError:
                                        # If JSON fails, try literal_eval for Python dict strings
                                        output_data = ast.literal_eval(result_str)

                                    if (
                                        isinstance(output_data, dict)
                                        and "output_data" in output_data
                                    ):
                                        benchmark_chunks = output_data[
                                            "output_data"
                                        ].get("benchmark_document_chunks", [])
                                        all_retrieved_chunks.extend(benchmark_chunks)

                                        # Use the LLM context string as tool result
                                        llm_context = output_data["output_data"].get(
                                            "llm_context_string", ""
                                        )
                                        tool_result_content = (
                                            f"Retrieved documents:\n{llm_context}"
                                        )
                                    else:
                                        tool_result_content = result_str
                                except (json.JSONDecodeError, ValueError, SyntaxError):
                                    tool_result_content = result_str

                            # Handle return_final_answer tool result
                            elif tool_name == "return_final_answer":
                                result_str = tool_call_result.result
                                try:
                                    import json
                                    import ast

                                    # First try JSON parsing
                                    try:
                                        output_data = json.loads(result_str)
                                    except json.JSONDecodeError:
                                        # If JSON fails, try literal_eval for Python dict strings
                                        output_data = ast.literal_eval(result_str)

                                    if (
                                        isinstance(output_data, dict)
                                        and "output_data" in output_data
                                    ):
                                        final_generation = output_data[
                                            "output_data"
                                        ].get("final_answer", "")
                                    else:
                                        final_generation = result_str
                                except (json.JSONDecodeError, ValueError, SyntaxError):
                                    final_generation = result_str
                                tool_call_successful = True
                                break

                            else:
                                tool_result_content = tool_call_result.result

                            # Add tool result to conversation
                            messages.append(reply)
                            messages.append(
                                ChatMessage.from_tool(
                                    tool_result=tool_result_content,
                                    origin=tool_call,
                                    error=False,
                                )
                            )
                            tool_call_successful = True
                        else:
                            # Tool execution had an error
                            error_msg = (
                                tool_call_result.result
                                if tool_call_result
                                else "Unknown tool error"
                            )
                            messages.append(reply)
                            messages.append(
                                ChatMessage.from_tool(
                                    tool_result=error_msg, origin=tool_call, error=True
                                )
                            )
                    else:
                        # No tool messages returned
                        error_msg = f"No result from tool '{tool_name}'"
                        messages.append(reply)
                        messages.append(
                            ChatMessage.from_tool(
                                tool_result=error_msg, origin=tool_call, error=True
                            )
                        )

                except Exception as e:
                    # Tool execution failed
                    error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                    messages.append(reply)
                    messages.append(
                        ChatMessage.from_tool(
                            tool_result=error_msg, origin=tool_call, error=True
                        )
                    )

            # If return_final_answer was called successfully, break
            if tool_name == "return_final_answer" and tool_call_successful:
                break

            # If unknown tool handling caused break, exit
            if final_generation in ["##unknown_tool_called"]:
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
