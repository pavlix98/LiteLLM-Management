"""Feature for checking available LiteLLM models."""

from collections.abc import Sequence
from time import monotonic

from openai import OpenAIError

from litellm_management.cli_ui import CliConsole, FeatureResultSummary, ModelTestResultRow
from litellm_management.config import LiteLlmConfigLoader, MissingLiteLlmApiTokenError
from litellm_management.features.base import Feature, FeatureDefinition
from litellm_management.litellm_client import AvailableModel
from litellm_management.litellm_client import LiteLlmClient


class TestAvailableModelsFeature(Feature):
    """Check which models are available through LiteLLM."""

    def __init__(self, console: CliConsole | None = None) -> None:
        super().__init__(
            FeatureDefinition(
                flag="--test-available-models",
                description="Test which LiteLLM models are available.",
            )
        )
        self._console = console or CliConsole()

    def run(self) -> int:
        """Run the available-models check."""
        start_time = monotonic()

        try:
            config = LiteLlmConfigLoader().load()
            self._console.show_feature_header(
                feature_name="Test available models",
                endpoint_url=config.base_url,
            )

            with self._console.status("Getting available models"):
                models = LiteLlmClient(config).list_models()
        except MissingLiteLlmApiTokenError as error:
            self._console.show_error(str(error))
            return 1
        except OpenAIError as error:
            self._console.show_error(f"LiteLLM API error: {error}")
            return 1

        if not models:
            self._console.show_empty("No models are available.")
            return 0

        self._console.show_success(f"Found {len(models)} models")
        result_rows = self._test_models(models)
        duration_seconds = monotonic() - start_time
        self._console.show_results(
            FeatureResultSummary(
                available_count=len(models),
                failed_count=0,
                duration_seconds=duration_seconds,
            )
        )
        self._console.show_model_results_table(result_rows)

        return 0

    def _test_models(self, models: Sequence[AvailableModel]) -> list[ModelTestResultRow]:
        self._console.test_models_progress([model.id for model in models])
        return [
            ModelTestResultRow(
                model_name=model.id,
                status="skipped",
                response="Not tested yet",
            )
            for model in models
        ]
