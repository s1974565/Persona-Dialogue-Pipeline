import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

########## Common ##########

DEEPINFRA_HTTPS_ENDPOINT = os.getenv("DEEPINFRA_HTTPS_ENDPOINT")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

########## Completion ##########

DEEPINFRA_V1_BASE_URL = DEEPINFRA_HTTPS_ENDPOINT.rstrip("/") + "/v1/openai" if DEEPINFRA_HTTPS_ENDPOINT else None
