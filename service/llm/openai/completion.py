import openai
import asyncio
import contextlib
import json
import copy
import pydantic
from typing import Type, Any
from collections.abc import Awaitable, Callable

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
    COMPLETION_API_VERSION,
    COMPLETION_ESCALATION_CHAIN,
    COMPLETION_CONCURRENCY_LIMITS,
    COMPLETION_REASONING_EFFORT,
    COMPLETION_VERBOSITY,
    NON_RETRYABLE_ERRORS,
    AZURE_HTTPS_ENDPOINT,
    AZURE_API_KEY,
)


# Process-wide semaphores for models with a configured concurrency cap. asyncio.Semaphore
# binds to an event loop on first use, not at construction, so building these at import time
# is safe as long as the entire process drives them from one loop (which is the case here —
# the runner calls `asyncio.run` once).
_MODEL_CONCURRENCY_SEMAPHORES: dict[CompletionModel, asyncio.Semaphore] = {
    model: asyncio.Semaphore(limit) for model, limit in COMPLETION_CONCURRENCY_LIMITS.items()
}


class Completion(CompletionSchema):
    def __init__(self):
        super().__init__()

        self.azure_client = openai.AsyncAzureOpenAI(
            azure_endpoint=AZURE_HTTPS_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=COMPLETION_API_VERSION,
        )

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENAI

    def _build_api_messages(self, dialogue_history: list[dict]) -> tuple[list[dict], str]:
        api_messages = []

        for entry in dialogue_history:
            role = entry["role"]
            content = entry["content"]

            if isinstance(content, str):
                api_messages.append({"role": role, "content": content})
                continue

            # Transform content blocks to OpenAI format.
            transformed_content = []
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    transformed_content.append(
                        {
                            "type": "output_text" if role == DialogueRole.ASSISTANT else "input_text",
                            "text": block["text"],
                        }
                    )
                elif block_type == "image":
                    transformed_content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{block['base64_image']}",
                        }
                    )
                else:
                    raise ValueError(f"Unknown dialogue block type: {block_type}")

            api_messages.append({"role": role, "content": transformed_content})

        return api_messages, COMPLETION_SYSTEM_INSTRUCTIONS

    async def _run_completion_with_retries(
        self,
        base_message: dict[str, Any],
        model: CompletionModel,
        request_label: str,
        request_context: dict[str, Any],
        attempt_call: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        """
        Run a single completion request with retries and exponential backoff.
        :param base_message: Base request payload.
        :param model: Model used for the request.
        :param request_label: Short label for the request.
        :param request_context: Request identifier.
        :param attempt_call: Coroutine performing one API attempt and returning the response.
        :return: The response from the first successful attempt.
        """
        number_of_attempts = 0
        response = None
        error_type = ""
        while number_of_attempts < COMPLETION_ERROR_MAXIMUM_TRY:
            # OpenAI library internally re-formats the given message. For example, text_format -> text.format
            message = copy.deepcopy(base_message)
            try:
                response = await attempt_call(message)
                break

            except openai.APIError as exception:
                error_type = type(exception).__name__
                if isinstance(exception, NON_RETRYABLE_ERRORS):
                    raise LLMAPIError(
                        f"Non-retryable error from {request_label} API.",
                        {
                            "error_type": error_type,
                            "status_code": getattr(exception, "status_code", None),
                            "message": str(exception),
                            "model": model,
                            **request_context,
                        },
                    ) from exception
                await asyncio.sleep(exponential(RATELIMIT_WAIT_EXPONENTIAL_COEFFICIENT, RATELIMIT_WAIT_EXPONENTIAL_BASE, number_of_attempts))

            except pydantic.ValidationError:
                error_type = "ValidationError"
                response = None

            finally:
                number_of_attempts += 1

        if response is None:
            raise LLMAPIError(
                f"The {request_label} request has exceeded the maximum number of retries.",
                {"error_type": error_type, "model": model, **request_context},
            )

        return response

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
        base_message: dict[str, Any] = {
            "model": model,
            "input": api_messages,
        }

        if system_instructions:
            base_message["instructions"] = system_instructions

        base_message["reasoning"] = {"effort": self._resolve_reasoning_effort(model, COMPLETION_REASONING_EFFORT[model], use_reasoning)}
        base_message["text"] = {"verbosity": COMPLETION_VERBOSITY[model]}

        if response_format:
            base_message["text_format"] = response_format

        def record_usage(api_response: Any) -> None:
            usage = getattr(api_response, "usage", None)
            if usage_sink is None or usage is None:
                return
            usage_sink.update(
                {
                    "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                    "cached_input_tokens": getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0,
                    "cache_write_tokens": 0,
                }
            )

        async def attempt_call(message: dict[str, Any]) -> Any:
            if response_format:
                api_response = await self.azure_client.responses.parse(**message)
                record_usage(api_response)
                parsed_response = api_response.output_parsed
                response_format.model_validate(parsed_response)
                return parsed_response
            api_response = await self.azure_client.responses.create(**message)
            record_usage(api_response)
            return api_response.output_text

        # Hold the per-model concurrency slot across the entire retry window so a single
        # logical call counts as one tenant against the cap, not one per network attempt.
        concurrency_slot = _MODEL_CONCURRENCY_SEMAPHORES.get(model) or contextlib.nullcontext()
        async with concurrency_slot:
            return await self._run_completion_with_retries(
                base_message=base_message,
                model=model,
                request_label="completion",
                request_context={"prompt": base_message},
                attempt_call=attempt_call,
            )
