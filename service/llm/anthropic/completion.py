import anthropic
import asyncio
import json
import copy
import pydantic
from typing import Type, Any

from service.configurations import DialogueRole
from service.utilities import exponential
from service.llm.configurations import (
    LLMAPIError,
    LLMProvider,
    CompletionModel,
    CompletionResponseFormat,
    ReasoningEffort,
    RATELIMIT_WAIT_EXPONENTIAL_COEFFICIENT,
    RATELIMIT_WAIT_EXPONENTIAL_BASE,
    COMPLETION_SYSTEM_INSTRUCTIONS,
    COMPLETION_ERROR_MAXIMUM_TRY,
)
from service.llm.completion_schema import CompletionSchema
from . import configurations as anthropic_configurations
from .configurations import (
    ANTHROPIC_API_KEY,
    SYSTEM_INSTRUCTION_PREFIX,
    COMPLETION_ESCALATION_CHAIN,
    COMPLETION_MAX_TOKENS,
    COMPLETION_REASONING_EFFORT,
    NON_RETRYABLE_ERRORS,
)


class Completion(CompletionSchema):
    def __init__(self):
        super().__init__()

        self.is_developer_role_allowed = False
        self.client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC

    def _build_api_messages(self, dialogue_history: list[dict]) -> tuple[list[dict], str]:
        api_messages = []
        developer_instructions_parts = []

        if COMPLETION_SYSTEM_INSTRUCTIONS:
            developer_instructions_parts.append(COMPLETION_SYSTEM_INSTRUCTIONS)

        for entry in dialogue_history:
            role = entry["role"]
            content = entry["content"]

            # Anthropic does not support developer (system) roles. Initial developer messages are folded into the system instructions,
            # and mid-dialogue developer messages are re-routed to user role with special system prefix.
            if role in [DialogueRole.DEVELOPER, DialogueRole.SYSTEM]:
                if len(api_messages) == 0:
                    for block in content:
                        if block.get("type") == "text":
                            developer_instructions_parts.append(block["text"])
                    continue

                transformed_content = []
                for block in content:
                    if block.get("type") == "text":
                        transformed_content.append(
                            {
                                "type": "text",
                                "text": f"{SYSTEM_INSTRUCTION_PREFIX} {block['text']}\n\n",
                            }
                        )
                    else:
                        raise ValueError(f"Unsupported block type for mid-dialogue developer (system) message: {block.get('type')}")

                api_messages.append({"role": DialogueRole.USER, "content": transformed_content})
                continue

            # Transform content blocks to Anthropic format.
            transformed_content = []
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    transformed_content.append(
                        {
                            "type": "text",
                            "text": block["text"],
                        }
                    )
                elif block_type == "image":
                    transformed_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": block["base64_image"],
                            },
                        }
                    )
                else:
                    raise ValueError(f"Unknown dialogue block type: {block_type}")

            api_messages.append({"role": role, "content": transformed_content})

        return api_messages, "\n\n".join(developer_instructions_parts)

    async def create_single_history_response(
        self,
        dialogue_history: list[dict],
        model: CompletionModel | None = None,
        response_format: Type[CompletionResponseFormat] | None = None,
        use_reasoning: bool = True,
        usage_sink: dict | None = None,
    ) -> str | CompletionResponseFormat:
        if model is None:
            prompt = json.dumps(dialogue_history, sort_keys=True)
            prompt_count = await self.prompt_tracker.get_and_increment_count(prompt)

            tier_ceiling = 0
            for tier_model, tier_tries in COMPLETION_ESCALATION_CHAIN:
                tier_ceiling += tier_tries
                if prompt_count < tier_ceiling:
                    model = tier_model
                    break
            else:
                chain_summary = ", ".join(f"{tier_model}:{tier_tries}" for tier_model, tier_tries in COMPLETION_ESCALATION_CHAIN)
                error_message = f"Model escalation failed. The identical prompt has been tried too many times ({chain_summary})."
                error_details = {"prompt": prompt}
                raise LLMAPIError(error_message, error_details)
        else:
            if model.provider != self.provider:
                raise ValueError(f"{model} is not available in {self.provider}.")

        api_messages, system_instructions = self._build_api_messages(dialogue_history)
        cache_ttl = anthropic_configurations.COMPLETION_CACHE_TTL.value
        for message in api_messages:
            if message["role"] == DialogueRole.USER and message["content"]:
                message["content"][0]["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}
                break

        base_message: dict[str, Any] = {
            "model": model,
            "max_tokens": COMPLETION_MAX_TOKENS[model],
            "messages": api_messages,
            "cache_control": {"type": "ephemeral", "ttl": cache_ttl},
        }
        if system_instructions:
            base_message["system"] = [{"type": "text", "text": system_instructions, "cache_control": {"type": "ephemeral", "ttl": cache_ttl}}]

        resolved_reasoning = self._resolve_reasoning_effort(model, COMPLETION_REASONING_EFFORT[model], use_reasoning)
        base_message["thinking"] = {"type": "adaptive"} if use_reasoning else {"type": "disabled"}
        if resolved_reasoning != ReasoningEffort.DISABLED:
            base_message["output_config"] = {"effort": resolved_reasoning}

        if response_format:
            base_message["output_format"] = response_format

        number_of_attempts = 0
        response = None
        error_type = ""
        while number_of_attempts < COMPLETION_ERROR_MAXIMUM_TRY:
            message = copy.deepcopy(base_message)
            try:
                async with self.client.messages.stream(**message) as stream:
                    raw_response = await stream.get_final_message()

                if usage_sink is not None and raw_response.usage is not None:
                    usage = raw_response.usage
                    usage_sink.update(
                        {
                            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                            "cached_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                            "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
                        }
                    )

                text_parts = []
                for content_block in raw_response.content:
                    if content_block.type == "text":
                        text_parts.append(content_block.text)
                response = "".join(text_parts)

                if response_format:
                    response = response_format.model_validate_json(response)
                break

            except anthropic.APIError as exception:
                error_type = type(exception).__name__
                if isinstance(exception, NON_RETRYABLE_ERRORS):
                    raise LLMAPIError(
                        "Non-retryable error from completion API.",
                        {
                            "error_type": error_type,
                            "status_code": getattr(exception, "status_code", None),
                            "message": str(exception),
                            "model": model,
                            "prompt": base_message,
                        },
                    ) from exception
                await asyncio.sleep(exponential(RATELIMIT_WAIT_EXPONENTIAL_COEFFICIENT, RATELIMIT_WAIT_EXPONENTIAL_BASE, number_of_attempts))

            except pydantic.ValidationError:
                error_type = "ValidationError"
                response = None

            finally:
                number_of_attempts += 1

        if response is None:
            error_message = "The current completion request has exceeded the maximum number of retries."
            error_details = {"error_type": error_type, "model": model, "prompt": base_message}
            raise LLMAPIError(error_message, error_details)

        return response
