import os
from enum import StrEnum
from pathlib import Path
from dotenv import load_dotenv
import anthropic

from service.llm.configurations import CompletionModel, ReasoningEffort

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

########## Common ##########

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

SYSTEM_INSTRUCTION_PREFIX = "[SYSTEM]"

NON_RETRYABLE_ERRORS = (
    anthropic.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.UnprocessableEntityError,
)

########## Completion ##########

# Check: https://platform.claude.com/docs/en/about-claude/models/overview
COMPLETION_ESCALATION_CHAIN: list[tuple[CompletionModel, int]] = [
    (CompletionModel.CLAUDE_HAIKU_4_5, 10),
    (CompletionModel.CLAUDE_SONNET_4_6, 5),
]

COMPLETION_MAX_TOKENS: dict[CompletionModel, int] = {
    CompletionModel.CLAUDE_HAIKU_4_5: 64000,
    CompletionModel.CLAUDE_SONNET_4_6: 64000,
}

COMPLETION_REASONING_EFFORT: dict[CompletionModel, list[ReasoningEffort]] = {
    CompletionModel.CLAUDE_HAIKU_4_5: [ReasoningEffort.DISABLED, ReasoningEffort.UNSUPPORTED],  # No effort parameter; reasoning-on unused.
    CompletionModel.CLAUDE_SONNET_4_6: [ReasoningEffort.LOW, ReasoningEffort.MEDIUM],
}


class CacheTTL(StrEnum):
    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"


COMPLETION_CACHE_TTL: CacheTTL = CacheTTL.FIVE_MINUTES

