import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from service.llm import (
    AnthropicCompletion,
    AzureCompletion,
    CompletionModel,
    DeepInfraCompletion,
    GoogleCompletion,
    LLMProvider,
    OpenAICompletion,
)

from configurations import CONVERSATION_CONCURRENCY, MODELS_WITHOUT_REASONING_TOGGLE

CompletionClient = OpenAICompletion | AnthropicCompletion | AzureCompletion | GoogleCompletion | DeepInfraCompletion
TaskResult = TypeVar("TaskResult")

_COMPLETION_CLASS: dict[LLMProvider, type[CompletionClient]] = {
    LLMProvider.OPENAI: OpenAICompletion,
    LLMProvider.ANTHROPIC: AnthropicCompletion,
    LLMProvider.AZURE: AzureCompletion,
    LLMProvider.GOOGLE: GoogleCompletion,
    LLMProvider.DEEPINFRA: DeepInfraCompletion,
}
_model_client_cache: dict[LLMProvider, CompletionClient] = {}


def resolve_model_client(model_name: str | CompletionModel, use_cached: bool = True) -> CompletionClient:
    model = CompletionModel(str(model_name).strip())
    provider = model.provider
    if provider not in _COMPLETION_CLASS:
        raise ValueError(f"Unknown provider: {provider}")
    completion_class = _COMPLETION_CLASS[provider]

    if not use_cached:
        return completion_class()

    if provider not in _model_client_cache:
        _model_client_cache[provider] = completion_class()
    return _model_client_cache[provider]


def reasoning_for(model: str | CompletionModel, run_reasoning: bool = True) -> bool:
    return False if CompletionModel(str(model)) in MODELS_WITHOUT_REASONING_TOGGLE else run_reasoning


async def gather_capped(thunks: list[Callable[[], Awaitable[TaskResult]]], limit: int = CONVERSATION_CONCURRENCY) -> list[TaskResult]:
    semaphore = asyncio.Semaphore(limit)

    async def _run(thunk: Callable[[], Awaitable[TaskResult]]) -> TaskResult:
        async with semaphore:
            return await thunk()

    return await asyncio.gather(*[_run(thunk) for thunk in thunks])
