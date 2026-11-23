import openai
import asyncio
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
from .configurations import (
    AZURE_V1_BASE_URL,
    AZURE_API_KEY,
    COMPLETION_REASONING_EFFORT,
    NON_RETRYABLE_ERRORS,
)


class Completion(CompletionSchema):
    def __init__(self, base_url: str = AZURE_V1_BASE_URL, api_key: str | None = AZURE_API_KEY):
        super().__init__()

        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.AZURE

    def _build_api_messages(self, dialogue_history: list[dict]) -> tuple[list[dict], str]:
        api_messages = []

        # Chat Completions uses a system-role message; prepend once if configured.
        if COMPLETION_SYSTEM_INSTRUCTIONS:
            api_messages.append({"role": DialogueRole.SYSTEM, "content": COMPLETION_SYSTEM_INSTRUCTIONS})

        for entry in dialogue_history:
            role = entry["role"]
            content = entry["content"]

            # Developer role is not supported by Azure v1 chat-completions API.
            if role == DialogueRole.DEVELOPER:
                role = DialogueRole.SYSTEM

            if isinstance(content, str):
                api_messages.append({"role": role, "content": content})
                continue

            # Transform content blocks to Chat Completions format.
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
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{block['base64_image']}"},
                        }
                    )
                else:
                    raise ValueError(f"Unknown dialogue block type: {block_type}")

            api_messages.append({"role": role, "content": transformed_content})

        # System instructions are embedded in api_messages, so the second return is intentionally empty.
        return api_messages, ""

    async def create_single_history_response(
        self,
        dialogue_history: list[dict],
        model: CompletionModel | None = None,
        response_format: Type[CompletionResponseFormat] | None = None,
        use_reasoning: bool = True,
        usage_sink: dict | None = None,
    ) -> str | CompletionResponseFormat:
        if model is None:
            raise ValueError("AzureCompletion requires explicit model argument; auto escalation is not available.")
        if model.provider != self.provider:
            raise ValueError(f"{model} is not available in {self.provider}.")

        api_messages, _system_instructions = self._build_api_messages(dialogue_history)
        base_message: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }

        options = COMPLETION_REASONING_EFFORT.get(model)
        if options is not None:
            resolved_reasoning = self._resolve_reasoning_effort(model, options, use_reasoning)
            if resolved_reasoning != ReasoningEffort.DISABLED:
                base_message["reasoning_effort"] = resolved_reasoning

        base_message.update(self._extra_completion_kwargs(model, use_reasoning))

        number_of_attempts = 0
        response = None
        error_type = ""
        while number_of_attempts < COMPLETION_ERROR_MAXIMUM_TRY:
            message = copy.deepcopy(base_message)
            try:
                if response_format:
                    raw_response = await self.client.beta.chat.completions.parse(
                        response_format=response_format,
                        **message,
                    )
                    response = raw_response.choices[0].message.parsed
                    response_format.model_validate(response)
                else:
                    raw_response = await self.client.chat.completions.create(**message)
                    response = raw_response.choices[0].message.content

                usage = getattr(raw_response, "usage", None)
                if usage_sink is not None and usage is not None:
                    usage_sink.update(
                        {
                            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                            "cached_input_tokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                            "cache_write_tokens": 0,
                        }
                    )
                break

            except openai.APIError as exception:
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
