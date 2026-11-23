import asyncio
import json
import copy
import pydantic
from typing import Type, Any

from google import genai
from google.genai import errors

from service.configurations import DialogueRole
from service.utilities import exponential
from service.llm.configurations import (
    LLMAPIError,
    LLMProvider,
    CompletionModel,
    CompletionResponseFormat,
    RATELIMIT_WAIT_EXPONENTIAL_COEFFICIENT,
    RATELIMIT_WAIT_EXPONENTIAL_BASE,
    COMPLETION_SYSTEM_INSTRUCTIONS,
    COMPLETION_ERROR_MAXIMUM_TRY,
)
from service.llm.completion_schema import CompletionSchema
from .configurations import (
    GOOGLE_API_KEY,
    SYSTEM_INSTRUCTION_PREFIX,
    COMPLETION_ESCALATION_CHAIN,
    COMPLETION_THINKING_LEVEL,
    NON_RETRYABLE_STATUS_CODES,
)


class Completion(CompletionSchema):
    def __init__(self):
        super().__init__()

        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.GOOGLE

    def _build_api_messages(self, dialogue_history: list[dict]) -> tuple[list[dict], str]:
        api_messages = []
        developer_instructions_parts = []

        if COMPLETION_SYSTEM_INSTRUCTIONS:
            developer_instructions_parts.append(COMPLETION_SYSTEM_INSTRUCTIONS)

        for entry in dialogue_history:
            role = entry["role"]
            content = entry["content"]

            # Gemini has no developer (system) role. Initial developer messages are folded into the system instruction,
            # and mid-dialogue developer messages are re-routed to the user role with a special system prefix.
            if role in [DialogueRole.DEVELOPER, DialogueRole.SYSTEM]:
                if len(api_messages) == 0:
                    for block in content:
                        if block.get("type") == "text":
                            developer_instructions_parts.append(block["text"])
                    continue

                transformed_parts = []
                for block in content:
                    if block.get("type") == "text":
                        transformed_parts.append({"text": f"{SYSTEM_INSTRUCTION_PREFIX} {block['text']}\n\n"})
                    else:
                        raise ValueError(f"Unsupported block type for mid-dialogue developer (system) message: {block.get('type')}")

                api_messages.append({"role": "user", "parts": transformed_parts})
                continue

            # Gemini uses "model" for assistant turns and "user" for everything else.
            gemini_role = "model" if role == DialogueRole.ASSISTANT else "user"

            transformed_parts = []
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    transformed_parts.append({"text": block["text"]})
                elif block_type == "image":
                    transformed_parts.append(
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": block["base64_image"],
                            }
                        }
                    )
                else:
                    raise ValueError(f"Unknown dialogue block type: {block_type}")

            api_messages.append({"role": gemini_role, "parts": transformed_parts})

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

        config: dict[str, Any] = {
            "thinking_config": {"thinking_level": self._resolve_reasoning_effort(model, COMPLETION_THINKING_LEVEL[model], use_reasoning)},
        }
        if system_instructions:
            config["system_instruction"] = system_instructions
        if response_format:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_format

        base_message: dict[str, Any] = {
            "model": model,
            "contents": api_messages,
            "config": config,
        }

        number_of_attempts = 0
        response = None
        error_type = ""
        while number_of_attempts < COMPLETION_ERROR_MAXIMUM_TRY:
            message = copy.deepcopy(base_message)
            try:
                raw_response = await self.client.aio.models.generate_content(**message)

                if usage_sink is not None and raw_response.usage_metadata is not None:
                    usage = raw_response.usage_metadata
                    usage_sink.update(
                        {
                            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                            "cached_input_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
                            "cache_write_tokens": 0,
                        }
                    )

                if response_format:
                    # Gemini 3 thinking turns may leave .parsed empty, so fall back to validating the raw JSON text.
                    parsed = raw_response.parsed
                    if parsed is not None:
                        response = response_format.model_validate(parsed)
                    else:
                        response = response_format.model_validate_json(raw_response.text)
                else:
                    response = raw_response.text
                break

            except errors.APIError as exception:
                error_type = type(exception).__name__
                if getattr(exception, "code", None) in NON_RETRYABLE_STATUS_CODES:
                    raise LLMAPIError(
                        "Non-retryable error from completion API.",
                        {
                            "error_type": error_type,
                            "status_code": getattr(exception, "code", None),
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
