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
    # As an ablation, let's try the simplest possible reward function: just give
    # 1 point for a correct answer, and 0 for anything else. Otherwise, we'll do something
    # more complex.
    if rubric.answer_correct:
        return 1
    else:
        return 0

    traj.logs.append(f"Rubric: {rubric}")
    traj.logs.append("Rubric not handled properly")
    raise ValueError("Rubric is not handled properly")


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

    traj.messages_and_choices = [
        {"role": "user", "content": scenario.query},
        # {"role": "assistant", "content": "<think>"},
    ]

    llm_response = None
    final_answer = None

    client = model.openai_client()
    model_name = model.config.base_model
    verifier = getattr(verifiers, model.config.verifier)

    while True:
        if rubric.num_tool_calls > model.config.max_tool_calls:
            break
        if rubric.completion_tokens > model.config.max_tokens:
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
                # max_tokens=1000 - rubric.completion_tokens,
                tools=tools,
                tool_choice="auto",
                extra_body=extra_body,
            )
        except Exception as e:
            llm_response = None
            break
        rubric.prompt_tokens += llm_response.usage.prompt_tokens  # type: ignore
        rubric.completion_tokens += llm_response.usage.completion_tokens  # type: ignore

        choice = llm_response.choices[0] if llm_response else None  # type: ignore
        if choice:
            traj.messages_and_choices.append(choice)  # type: ignore

        tool_calls = getattr(choice.message, "tool_calls", None) if choice else None
        if not tool_calls:
            rubric.has_answer = False
            break
        try:
            tool_call = tool_calls[0]
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            tool_id = getattr(tool_call, "id", None)
            rubric.num_tool_calls += 1

            # Early check for empty arguments
            if not tool_args_str or not tool_args_str.strip():
                rubric.has_answer = False
                break
        except Exception as e:
            print(f"Error parsing tool call: {e}")
            break

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

    reward = calculate_reward(getattr(model, "config", None), rubric, traj)
    traj.reward = reward
    traj.metrics = rubric.to_metrics()
    return traj.finish()


if __name__ == "__main__":
    pass
