"""Feature for checking available LiteLLM models."""

from litellm_management.features.base import Feature, FeatureDefinition


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
        print("Available model testing is not implemented yet.")
        return 0
