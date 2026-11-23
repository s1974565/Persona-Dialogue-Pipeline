import os
from pathlib import Path
from dotenv import load_dotenv
import openai

from service.llm.configurations import CompletionModel, ReasoningEffort

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

########## Common ##########

AZURE_HTTPS_ENDPOINT = os.getenv("AZURE_HTTPS_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

NON_RETRYABLE_ERRORS = (
    openai.BadRequestError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.UnprocessableEntityError,
)

########## Completion ##########

COMPLETION_API_VERSION = "2025-04-01-preview"

COMPLETION_ESCALATION_CHAIN: list[tuple[CompletionModel, int]] = [
    (CompletionModel.GPT_5_4_MINI, 10),
    (CompletionModel.GPT_5_5, 5),
]

COMPLETION_CONCURRENCY_LIMITS: dict[CompletionModel, int] = {}

COMPLETION_REASONING_EFFORT: dict[CompletionModel, list[ReasoningEffort]] = {
    CompletionModel.GPT_5_4_MINI: [ReasoningEffort.NONE, ReasoningEffort.MEDIUM],
    CompletionModel.GPT_5_5: [ReasoningEffort.NONE, ReasoningEffort.MEDIUM],
}

# "low", "medium", or "high".
COMPLETION_VERBOSITY: dict[CompletionModel, str] = {
    CompletionModel.GPT_5_4_MINI: "low",
    CompletionModel.GPT_5_5: "low",
}
