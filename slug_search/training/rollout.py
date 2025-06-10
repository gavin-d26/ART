import asyncio
import importlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime

from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import art
from art import Trajectory
from slug_search.training import verifiers
from slug_search.training.data_types import ProjectPolicyConfig, SearchQuery
from slug_search.training.search_tools import return_final_answer, search_documents


@dataclass
class SearchRubric:
    has_answer: bool = False
    answer_correct: bool = False
    num_tool_calls: int = 0
    num_successful_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ran_out_of_tool_calls: bool = False
    ran_out_of_tokens: bool = False

    def to_metrics(self) -> dict[str, float | int]:
        return {k: int(v) for k, v in asdict(self).items()}


openai_search_documents = convert_to_openai_tool(search_documents)
openai_return_final_answer = convert_to_openai_tool(return_final_answer)

tools = [openai_search_documents, openai_return_final_answer]


async def get_tool_result(tool_name: str, tool_args: dict) -> str:
    if tool_name == "search_documents":
        return await search_documents(tool_args["search_query"])
    elif tool_name == "return_final_answer":
        return await return_final_answer(tool_args["answer"])
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def calculate_reward(
    policy_config: ProjectPolicyConfig, rubric: SearchRubric, traj: Trajectory
) -> float:
    # use simple reward function
    # 1.0 for correct answer
    # 0.0 for incorrect answer

    # --- Start of Tunable Reward Hyperparameters ---
    reward_for_correct_answer: float = 1.0

    # To reduce reward sparsity, add a bonus for each successful tool call
    bonus_per_successful_tool_call: float = 0.1

    # Penalty for each failed tool call
    # The original function implicitly applied -1.0 for each failed tool call
    # by using `total_reward += rubric.num_successful_tool_calls - rubric.num_tool_calls`.
    # Making it explicit allows for better tuning.
    penalty_per_failed_tool_call: float = (
        0.0  # Example: less severe than original implicit -1
    )

    # Penalty factor if the agent answers incorrectly without using all available tool calls
    # Original factor was -0.5; adjusting this can change exploration/exploitation balance.
    penalty_factor_early_incorrect_stop: float = 0.0

    # Penalties for resource exhaustion
    penalty_ran_out_of_tool_calls: float = 0.0
    penalty_ran_out_of_tokens: float = 0.0
    # --- End of Tunable Reward Hyperparameters ---

    total_reward = 0.0

    if rubric.answer_correct:
        total_reward += reward_for_correct_answer
    # No direct penalty for just being incorrect, but the agent misses the +1.0 reward.
    # Penalties below will further shape behavior for incorrect trajectories.

    # Add bonus for successful tool calls (this is key for reducing sparsity)
    total_reward += rubric.num_successful_tool_calls * bonus_per_successful_tool_call

    # Penalize failed tool calls
    # This replaces the original `total_reward += rubric.num_successful_tool_calls - rubric.num_tool_calls`
    num_failed_tool_calls = rubric.num_tool_calls - rubric.num_successful_tool_calls
    total_reward += num_failed_tool_calls * penalty_per_failed_tool_call

    # Penalty for stopping early with an incorrect answer when resources were still available
    if (
        not rubric.answer_correct
        and rubric.num_tool_calls < policy_config.max_tool_calls
        and not rubric.ran_out_of_tool_calls
        and not rubric.ran_out_of_tokens
    ):
        unused_tool_calls = (policy_config.max_tool_calls - 1) - rubric.num_tool_calls
        total_reward += unused_tool_calls * penalty_factor_early_incorrect_stop

    if rubric.ran_out_of_tool_calls:
        total_reward += penalty_ran_out_of_tool_calls
    if rubric.ran_out_of_tokens:
        total_reward += penalty_ran_out_of_tokens

    return total_reward


async def rollout(
    model: art.TrainableModel,
    scenario: SearchQuery,
) -> Trajectory:
    rubric = SearchRubric()
    traj = Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"scenario_id": scenario.id},
    )
    # assert isinstance(model.config, ProjectPolicyConfig)

    traj.messages_and_choices = []

    if model.config.system_prompt:
        traj.messages_and_choices.append(
            {"role": "system", "content": model.config.system_prompt}
        )

    traj.logs.append(f"Answer: {scenario.answer}")
    traj.messages_and_choices.append({"role": "user", "content": scenario.query})

    llm_response = None
    final_answer = None

    client = model.openai_client()
    model_name = model.config.base_model
    verifier = getattr(verifiers, model.config.verifier)

    while True:
        if rubric.num_tool_calls > model.config.max_tool_calls:
            rubric.ran_out_of_tool_calls = True
            break
        if rubric.completion_tokens > model.config.max_tokens:
            rubric.ran_out_of_tokens = True
            break

        extra_body = {
            "add_generation_prompt": True,
            "continue_final_message": False,
        }
        if getattr(model, "config", None) and getattr(model.config, "custom_chat_template", None): # fmt: skip
            extra_body["chat_template"] = model.config.custom_chat_template
        try:
            llm_response = await client.chat.completions.create(
                model=model_name,
                messages=traj.messages(),
                temperature=model.config.temperature,
                top_p=model.config.top_p,
                # max_tokens=1000 - rubric.completion_tokens,
                tools=tools,
                tool_choice="auto",
                extra_body=extra_body,
            )
        except Exception as e:
            llm_response = None
            break

        if rubric.prompt_tokens == 0:
            rubric.prompt_tokens = llm_response.usage.prompt_tokens  # type: ignore

        choice = llm_response.choices[0] if llm_response else None  # type: ignore
        if choice:
            traj.messages_and_choices.append(choice)  # type: ignore

        tool_calls = getattr(choice.message, "tool_calls", None) if choice else None

        if not tool_calls:
            traj.messages_and_choices.append(
                {
                    "role": "tool",
                    "tool_call_id": "no_tool_call",
                    "name": "no_tool_call",
                    "content": "Error parsing tool call",
                }
            )
            rubric.num_tool_calls += 1
            continue
        try:
            tool_call = tool_calls[0]
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            tool_id = getattr(tool_call, "id", None)
            rubric.num_tool_calls += 1

            # Early check for empty arguments
            if not tool_args_str or not tool_args_str.strip():
                raise Exception("Error parsing tool call arguments")
        except Exception as e:
            print(f"Error parsing tool call: {e}")
            traj.messages_and_choices.append(
                {
                    "role": "tool",
                    "tool_call_id": "no_tool_call",
                    "name": "no_tool_call",
                    "content": str(e),
                }
            )
            rubric.num_tool_calls += 1
            continue

        if tool_name == "search_documents":
            # At this point, tool_args_str is guaranteed to be non-empty and non-whitespace
            try:
                tool_args = json.loads(tool_args_str)
                tool_result = await get_tool_result(tool_name, tool_args)
                rubric.num_successful_tool_calls += 1
            except Exception as e:
                tool_result = f"Error calling tool {tool_name}: {e}"
            traj.messages_and_choices.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": tool_result,
                }
            )
            continue
        elif tool_name == "return_final_answer":
            try:
                tool_args = json.loads(tool_args_str)
                answer = tool_args.get("answer")
                if answer is not None:
                    final_answer = answer
                    rubric.has_answer = True
                    rubric.answer_correct = verifier(final_answer, scenario.answer)
                    rubric.num_successful_tool_calls += 1
                else:
                    rubric.has_answer = False
            except Exception as e:
                rubric.has_answer = False
            # Do not append return_final_answer tool call to messages
            break
        else:
            rubric.has_answer = False
            tool_result = f"Unknown tool called: {tool_name}"
            traj.messages_and_choices.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": tool_result,
                }
            )
            break
    if llm_response:
        rubric.completion_tokens = (
            llm_response.usage.prompt_tokens
            + llm_response.usage.completion_tokens
            - rubric.prompt_tokens
        )  # completion tokens + tool result tokens
    else:
        rubric.completion_tokens = 0
    reward = calculate_reward(getattr(model, "config", None), rubric, traj)
    traj.reward = reward
    traj.metrics = rubric.to_metrics()
    return traj.finish()


if __name__ == "__main__":
    pass
