"""LiteLLM OpenAI-compatible API client."""

from time import monotonic

from openai import OpenAI
from pydantic import BaseModel, Field

from litellm_management.config import LiteLlmConfig


# LiteLLM model IDs listed here are tested through the Responses API.
# All other models use Chat Completions by default.
RESPONSES_API_MODEL_IDS = frozenset(
    {
        "GPT-5.5",
        "GPT-5.5 Pro",
        "gpt-5.5",
        "gpt-5.5-pro",
    }
)


class AvailableModel(BaseModel):
    """A model available through LiteLLM."""

    id: str = Field(min_length=1)


class LongGenerationResult(BaseModel):
    """Measured result of a long streamed Responses API generation."""

    model_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    ttfb_seconds: float | None = Field(default=None, ge=0)
    total_seconds: float = Field(ge=0)
    event_count: int = Field(ge=0)
    output_text: str

    @property
    def character_count(self) -> int:
        """Return the number of output characters."""
        return len(self.output_text)

    @property
    def word_count(self) -> int:
        """Return the approximate number of whitespace-delimited output words."""
        return len(self.output_text.split())


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

    def generate_long_response(
        self,
        model_id: str,
        prompt: str,
        timeout_seconds: float,
    ) -> LongGenerationResult:
        """Generate a long streamed response and measure response timings."""
        start_time = monotonic()
        ttfb_seconds: float | None = None
        event_count = 0
        output_parts: list[str] = []

        stream = self._client.responses.create(
            model=model_id,
            input=prompt,
            stream=True,
            timeout=timeout_seconds,
        )

        for event in stream:
            event_count += 1
            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if isinstance(delta, str) and delta != "":
                    if ttfb_seconds is None:
                        ttfb_seconds = monotonic() - start_time
                    output_parts.append(delta)

            if event_type == "response.error":
                message = getattr(
                    event,
                    "message",
                    "Unknown Responses API stream error.",
                )
                raise RuntimeError(str(message))

        return LongGenerationResult(
            model_id=model_id,
            prompt=prompt,
            ttfb_seconds=ttfb_seconds,
            total_seconds=monotonic() - start_time,
            event_count=event_count,
            output_text="".join(output_parts),
        )

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
