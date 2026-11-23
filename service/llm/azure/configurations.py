import os
from pathlib import Path
from dotenv import load_dotenv
import openai

from service.llm.configurations import CompletionModel, ReasoningEffort


########## Common ##########

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

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

# Azure OpenAI v1 path exposes an OpenAI chat-completion endpoint used for non-first-party deployments.
AZURE_V1_BASE_URL = AZURE_HTTPS_ENDPOINT.rstrip("/") + "/openai/v1" if AZURE_HTTPS_ENDPOINT else None

COMPLETION_REASONING_EFFORT: dict[CompletionModel, list[ReasoningEffort]] = {
    CompletionModel.DEEPSEEK_V4_FLASH: [ReasoningEffort.DISABLED, ReasoningEffort.MEDIUM],  # "low"/"medium" alias to "high"; off omits the param.
}
