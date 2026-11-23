from enum import StrEnum
from typing import TypeVar
from pydantic import BaseModel

from service.utilities import now


########## Common ##########

# RateLimitError wait time = coefficient * (base ** number of attempts)
RATELIMIT_WAIT_EXPONENTIAL_COEFFICIENT = 1
RATELIMIT_WAIT_EXPONENTIAL_BASE = 2


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GOOGLE = "google"
    DEEPINFRA = "deepinfra"


class LLMAPIError(Exception):
    """
    It is triggered when LLM API requests fail after all recovery attempts are made.
    """

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details if details is not None else {}
        self.timestamp = now()

    def __str__(self):
        error_message = super().__str__()
        if self.details:
            for key, value in self.details.items():
                error_message += f"\n{key}: {value}"

        return error_message


########## Completion ##########

COMPLETION_SYSTEM_INSTRUCTIONS = ""
COMPLETION_ERROR_MAXIMUM_TRY = 5

CompletionResponseFormat = TypeVar("CompletionResponseFormat", bound=BaseModel)


class CompletionModel(StrEnum):
    # OpenAI
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_5 = "gpt-5.5-1"

    # Anthropic
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"

    # Google
    GEMINI_3_5_FLASH = "gemini-3.5-flash"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"

    # Azure
    DEEPSEEK_V4_FLASH = "DeepSeek-V4-Flash"

    # DeepInfra
    QWEN_3_5_397B_A17B = "Qwen/Qwen3.5-397B-A17B"

    @property
    def provider(self) -> LLMProvider:
        _provider_mapping = {
            CompletionModel.GPT_5_4_MINI: LLMProvider.OPENAI,
            CompletionModel.GPT_5_5: LLMProvider.OPENAI,
            CompletionModel.CLAUDE_HAIKU_4_5: LLMProvider.ANTHROPIC,
            CompletionModel.CLAUDE_SONNET_4_6: LLMProvider.ANTHROPIC,
            CompletionModel.GEMINI_3_5_FLASH: LLMProvider.GOOGLE,
            CompletionModel.GEMINI_3_1_FLASH_LITE: LLMProvider.GOOGLE,
            CompletionModel.DEEPSEEK_V4_FLASH: LLMProvider.AZURE,
            CompletionModel.QWEN_3_5_397B_A17B: LLMProvider.DEEPINFRA,
        }
        return _provider_mapping[self]


class ReasoningEffort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
    XHIGH = "xhigh"
    DISABLED = "disabled"  # Reasoning off without an effort level.
    UNSUPPORTED = "unsupported"  # State not allowed for the model; requesting it raises ValueError.
