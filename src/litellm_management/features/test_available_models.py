"""Feature for checking available LiteLLM models."""

from openai import OpenAIError

from litellm_management.config import LiteLlmConfigLoader, MissingLiteLlmApiTokenError
from litellm_management.features.base import Feature, FeatureDefinition
from litellm_management.litellm_client import LiteLlmClient


class TestAvailableModelsFeature(Feature):
    """Check which models are available through LiteLLM."""

    def __init__(self) -> None:
        super().__init__(
            FeatureDefinition(
                flag="--test-available-models",
                description="Test which LiteLLM models are available.",
            )
        )

    def run(self) -> int:
        """Run the available-models check."""
        print("Getting available models...")

        try:
            config = LiteLlmConfigLoader().load()
            print(f"Using LiteLLM URL: {config.base_url}")
            models = LiteLlmClient(config).list_models()
        except MissingLiteLlmApiTokenError as error:
            print(f"Configuration error: {error}")
            return 1
        except OpenAIError as error:
            print(f"LiteLLM API error: {error}")
            return 1

        if not models:
            print("No models are available.")
            return 0

        print("Available models:")
        for model in models:
            print(f"- {model.id}")

        return 0
