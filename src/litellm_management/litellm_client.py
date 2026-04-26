"""LiteLLM OpenAI-compatible API client."""

from openai import OpenAI
from pydantic import BaseModel, Field

from litellm_management.config import LiteLlmConfig


# LiteLLM model IDs listed here are tested through the Responses API.
# All other models use Chat Completions by default.
RESPONSES_API_MODEL_IDS = frozenset(
    {
        "GPT-5.5 Pro",
        "gpt-5.5-pro",
    }
)


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

    def ask_model(self, model_id: str, prompt: str) -> str:
        """Send a prompt to a model and return its text response."""
        if model_id in RESPONSES_API_MODEL_IDS:
            return self._ask_responses_model(model_id=model_id, prompt=prompt)

        return self._ask_chat_model(model_id=model_id, prompt=prompt)

    def _ask_chat_model(self, model_id: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        content = response.choices[0].message.content

        if content is None:
            return ""

        return content

    def _ask_responses_model(self, model_id: str, prompt: str) -> str:
        response = self._client.responses.create(
            model=model_id,
            input=prompt,
        )
        return response.output_text
