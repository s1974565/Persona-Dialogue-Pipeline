from typing import Any

from service.llm.configurations import LLMProvider, CompletionModel
from service.llm.azure.completion import Completion as AzureCompletion
from .configurations import DEEPINFRA_V1_BASE_URL, DEEPINFRA_API_KEY


class Completion(AzureCompletion):
    def __init__(self):
        super().__init__(base_url=DEEPINFRA_V1_BASE_URL, api_key=DEEPINFRA_API_KEY)

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.DEEPINFRA

    def _extra_completion_kwargs(self, model: CompletionModel, use_reasoning: bool) -> dict[str, Any]:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": use_reasoning}}}
