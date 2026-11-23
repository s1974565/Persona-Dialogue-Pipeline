import os
from pathlib import Path
from dotenv import load_dotenv

from service.llm.configurations import CompletionModel, ReasoningEffort

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

########## Common ##########

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SYSTEM_INSTRUCTION_PREFIX = "[SYSTEM]"

# 429 (rate limit) is itself a 4xx, so non-retryability is decided by status code.
NON_RETRYABLE_STATUS_CODES: set[int] = {400, 401, 403, 404, 422}

########## Completion ##########

# Check: https://ai.google.dev/gemini-api/docs/models
COMPLETION_ESCALATION_CHAIN: list[tuple[CompletionModel, int]] = [
    (CompletionModel.GEMINI_3_1_FLASH_LITE, 10),
    (CompletionModel.GEMINI_3_5_FLASH, 5),
]

COMPLETION_THINKING_LEVEL: dict[CompletionModel, list[ReasoningEffort]] = {
    CompletionModel.GEMINI_3_1_FLASH_LITE: [ReasoningEffort.MINIMAL, ReasoningEffort.MEDIUM],
    CompletionModel.GEMINI_3_5_FLASH: [ReasoningEffort.MINIMAL, ReasoningEffort.MEDIUM],
}
