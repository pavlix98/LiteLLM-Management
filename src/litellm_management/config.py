"""Application configuration."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


LITELLM_BASE_URL = "https://litellm.quanti.cz/"
LITELLM_API_TOKEN_ENV_NAME = "LITELLM_API_TOKEN"


class LiteLlmConfig(BaseModel):
    """Configuration for the LiteLLM OpenAI-compatible API."""

    base_url: str = Field(default=LITELLM_BASE_URL, min_length=1)
    api_token: str = Field(min_length=1)


class LiteLlmConfigLoader:
    """Loads LiteLLM configuration from local environment files."""

    def load(self) -> LiteLlmConfig:
        """Load and validate LiteLLM configuration."""
        load_dotenv()
        api_token = os.getenv(LITELLM_API_TOKEN_ENV_NAME)

        if api_token is None or api_token.strip() == "":
            raise MissingLiteLlmApiTokenError(LITELLM_API_TOKEN_ENV_NAME)

        return LiteLlmConfig(api_token=api_token)


class MissingLiteLlmApiTokenError(RuntimeError):
    """Raised when the LiteLLM API token is not configured."""

    def __init__(self, env_name: str) -> None:
        super().__init__(
            f"Missing {env_name}. Create a local .env file from .env.example and set the token."
        )

