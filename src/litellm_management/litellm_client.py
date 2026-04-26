"""LiteLLM OpenAI-compatible API client."""

from openai import OpenAI
from pydantic import BaseModel, Field

from litellm_management.config import LiteLlmConfig


class AvailableModel(BaseModel):
    """A model available through LiteLLM."""

    id: str = Field(min_length=1)


class LiteLlmClient:
    """Client for the LiteLLM OpenAI-compatible API."""

    def __init__(self, config: LiteLlmConfig) -> None:
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_token)

    def list_models(self) -> list[AvailableModel]:
        """List available models."""
        response = self._client.models.list()
        return [AvailableModel(id=model.id) for model in response.data]
