import art
from typing import List, Any
from art import Trajectory
from litellm import acompletion
import litellm
from slug_search.training.search_tools import (
    search_relevant_documents_top_3,
    search_relevant_documents_top_5,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm.caching.caching import LiteLLMCacheType, Cache
import json
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from litellm.types.utils import Choices, ModelResponse, Message
from dataclasses import asdict
from art.utils.litellm import convert_litellm_choice_to_openai
from dataclasses import dataclass
from art.utils import limit_concurrency
from datetime import datetime
from slug_search.training.data_types import ProjectPolicyConfig, SearchQuery
from tenacity import retry, stop_after_attempt


# tools


def return_final_answer(answer: str, sources: List[str] | None) -> str:
    """
    This function is used to return the final answer to the user's query.
    It should be called with the answer and the sources. If you cannot find the answer, you should return "I don't know" with an empty list of sources.

    Args:
        answer: (str) the answer to the user's query. If you cannot find the answer, you should return "I don't know" with an empty list of sources.
        sources: (list[str]) a list of message ids that are relevant to the query. Usually there will be only one.

    Returns:
        (str) the final answer to the user's query
    """
    ...


search_relevant_documents_top_3 = convert_to_openai_tool(
    search_relevant_documents_top_3
)
search_relevant_documents_top_5 = convert_to_openai_tool(
    search_relevant_documents_top_5
)


tools: list[ChatCompletionToolParam] = [
    search_relevant_documents_top_3,
    search_relevant_documents_top_5,
    convert_to_openai_tool(return_final_answer),
]  # type: ignore


def tool_response(response: Any, message: Message) -> ChatCompletionMessageParam:
    """Generate a response for a tool call.

    Args:
        response: The response from the tool
        message: The message that's being responded to

    Returns:
        A message that can be added to the conversation
    """
    if message.tool_calls:
        return {
            "role": "tool",
            "tool_call_id": message.tool_calls[0].id,
            "content": json.dumps(response),
        }
    else:
        return {
            "role": "user",
            "content": json.dumps(response),
        }


# ------------------------------------------------------------


# rubric and reward function
@dataclass
class SearchRubric:
    answer_correct: bool = False
    num_tool_calls: int = 0
    num_bad_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ran_out_of_tool_calls: bool = False

    def to_metrics(self) -> dict[str, float | int]:
        return {k: int(v) for k, v in asdict(self).items()}


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


# ------------------------------------------------------------


@retry(stop=stop_after_attempt(3))
async def determine_if_answer_is_correct(answer: str, query: SearchQuery) -> bool:
    # TODO: Implement this
    pass


# @retry(stop=stop_after_attempt(3))
@limit_concurrency(10, derive_key=lambda model, scenario, **kwargs: model.name)
async def rollout(
    model: art.Model,
    scenario: SearchQuery,
) -> Trajectory:
    rollout_start_time = datetime.now()
    rubric = SearchRubric()
    traj = Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"scenario_id": scenario.id},
    )
    assert isinstance(model.config, ProjectPolicyConfig)

    traj.messages_and_choices = [{"role": "user", "content": scenario.query}]

    llm_response: ModelResponse | None = None
    final_answer = None

    while True:
        rubric.num_tool_calls += 1

        if rubric.num_tool_calls > model.config.max_tool_calls:
            rubric.ran_out_of_tool_calls = True
            traj.logs.append("Ran out of tool calls")
            break

        litellm_model_name = model.config.litellm_model_name
        if litellm_model_name is None:
            litellm_model_name = f"hosted_vllm/{model.name}"

        async with traj.track_duration("llm_completion"):
            llm_response = await acompletion(
                model=litellm_model_name,
                base_url=model.inference_base_url,
                messages=traj.messages(),
                caching=not model.trainable,
                api_key=model.inference_api_key,
                max_completion_tokens=model.config.max_tokens,
                tools=tools if model.config.use_tools else None,
                tool_choice=(
                    "required"
                    if model.config.use_tools and not model.trainable
                    else None
                ),
            )  # type: ignore

        assert isinstance(llm_response, ModelResponse)
        rubric.prompt_tokens += llm_response.usage.prompt_tokens  # type: ignore
        rubric.completion_tokens += llm_response.usage.completion_tokens  # type: ignore
        choice = llm_response.choices[0]  # type: ignore
        assert isinstance(choice, Choices)

        # Our rollout is only set up to handle one tool call at a time, so just ignore any parallel tool calls.
        if choice.message.tool_calls is not None and len(choice.message.tool_calls) > 1:
            choice.message.tool_calls = choice.message.tool_calls[:1]
        if model.trainable:
            traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))
        else:
            traj.messages_and_choices.append(choice.message.to_dict())  # type: ignore

        if model.config.use_tools:
            tool_call = (
                choice.message.tool_calls[0].get("function")
                if choice.message.tool_calls
                else None
            )
            if tool_call is None:
                rubric.bad_tool_call_args = True
                break
            tool_name = tool_call["name"]
            try:
                tool_args = json.loads(tool_call["arguments"])
                assert isinstance(tool_args, dict)
            except Exception as e:
                rubric.bad_tool_call_args = True
                break
        else:
            raw_content = choice.message.content
            if raw_content is None:
                rubric.cant_parse_tool_call = True
                break
            start_index = raw_content.find("{")
            end_index = raw_content.rfind("}")
            if not (start_index != -1 and end_index != -1 and start_index < end_index):
                rubric.cant_parse_tool_call = True
                break
            json_str = raw_content[start_index : end_index + 1]

            try:
                tool_call = json.loads(json_str)
            except Exception as e:
                traj.logs.append(f"Error parsing tool call: {e}")
                rubric.cant_parse_tool_call = True
                break

            if "tool_args" not in tool_call:
                rubric.bad_tool_call_args = True
                traj.logs.append(f"Tool call missing tool_args: {tool_call}")
                break
            tool_name = tool_call.get("tool_name")
            tool_args = tool_call.get("tool_args")

        match tool_name:
            case "search_emails":
                try:
                    search_results = search_emails(
                        **tool_args,
                        inbox=scenario.inbox_address,
                    )
                    traj.messages_and_choices.append(
                        tool_response(
                            [asdict(r) for r in search_results],
                            choice.message,
                        )
                    )
                    for r in search_results:
                        if r.message_id == scenario.message_ids[0]:
                            rubric.ever_found_right_email = True
                except Exception as e:
                    rubric.bad_tool_call_args = True
                    traj.logs.append(f"Error searching emails: {e}")
                    break
            case "read_email":
                message_id_to_read = tool_args.get("message_id")
                if not isinstance(message_id_to_read, str):
                    rubric.bad_tool_call_args = True
                    break
                if message_id_to_read == scenario.message_ids[0]:
                    rubric.ever_read_right_email = True
                email_content = read_email(message_id_to_read)
                if email_content is None:
                    traj.messages_and_choices.append(
                        tool_response({"error": "Email not found"}, choice.message)
                    )
                    rubric.ever_tried_to_read_invalid_email = True
                else:
                    traj.messages_and_choices.append(
                        tool_response(email_content.model_dump(), choice.message)
                    )
            case "return_final_answer":
                final_answer = tool_args.get("answer")
                final_sources = tool_args.get("sources")

                if (
                    final_answer is None
                    or final_sources is None
                    or not isinstance(final_sources, list)
                ):
                    rubric.bad_tool_call_args = True
                    break

                if final_answer == "I don't know":
                    rubric.returned_i_dont_know = True
                else:
                    rubric.attempted_answer = True
                    async with traj.track_duration("determine_if_answer_is_correct"):
                        rubric.answer_correct = await determine_if_answer_is_correct(
                            final_answer, scenario
                        )
                    rubric.sources_correct = scenario.message_ids[0] in final_sources
                break
            case _:
                rubric.bad_tool_call_name = True
                break

    reward = calculate_reward(model.config, rubric, traj)
    traj.reward = reward
    traj.metrics = rubric.to_metrics()
    rollout_end_time = datetime.now()  # Record end time
    # Compute duration in seconds and add to metrics
    duration_seconds = (rollout_end_time - rollout_start_time).total_seconds()
    traj.metrics["duration"] = duration_seconds

    if model.config.log_to_openpipe and op_client is not None:
        try:
            await op_client.report(
                requested_at=rollout_start_time.timestamp() * 1000,
                received_at=rollout_end_time.timestamp() * 1000,
                req_payload={
                    "model": model.name,
                    "messages": traj.messages()[:-1],
                    "metadata": {
                        "type": "enron_rollout_final",
                        "reward": str(traj.reward),
                        **{k: str(v) for k, v in traj.metrics.items()},
                        **{k: str(v) for k, v in traj.metadata.items()},
                    },
                },
                resp_payload=llm_response,
                status_code=200,
            )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

    return traj.finish()


if __name__ == "__main__":
    from art_e.data.query_iterators import load_synthetic_queries
    from dotenv import load_dotenv
    import asyncio
    import yaml

    load_dotenv()

    traj = asyncio.run(
        rollout(
            art.Model(
                name="gpt-4o",
                project="email_agent",
                config=ProjectPolicyConfig(
                    log_to_openpipe=False,
                    litellm_model_name="openai/gpt-4o",
                    use_tools=True,
                ),
            ),
            load_synthetic_queries(split="test", limit=1)[0],
        )
    )
    print(yaml.dump(traj.for_logging()))
