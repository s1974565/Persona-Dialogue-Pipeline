import os
import zoneinfo
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


########## Common ##########

TIMEZONE = zoneinfo.ZoneInfo(os.getenv("BACKEND_TIMEZONE", "UTC"))

class DialogueRole(StrEnum):
    DEVELOPER = "developer"
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
