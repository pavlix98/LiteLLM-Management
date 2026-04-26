"""Feature for checking available LiteLLM models."""

import ast

from collections.abc import Sequence
from time import monotonic
from typing import Any

from openai import OpenAIError

from litellm_management.cli_ui import CliConsole, FeatureResultSummary, ModelTestResultRow
from litellm_management.config import LiteLlmConfigLoader, MissingLiteLlmApiTokenError
from litellm_management.features.base import Feature, FeatureDefinition
from litellm_management.litellm_client import AvailableModel
from litellm_management.litellm_client import LiteLlmClient


TEST_PROMPT = "Co víš o vesnici Kuničky? maximálně 2 věty"
MAX_RESPONSE_LENGTH = 500


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
                description="Lists LiteLLM models and sends a small chat prompt to each one.",
                endpoint_url=config.base_url,
                test_prompt=TEST_PROMPT,
            )

            with self._console.status("Getting available models"):
                client = LiteLlmClient(config)
                models = client.list_models()
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
        result_rows = self._test_models(client, models)
        failed_count = sum(1 for row in result_rows if row.status == "failed")
        available_count = len(result_rows) - failed_count
        duration_seconds = monotonic() - start_time
        self._console.show_results(
            FeatureResultSummary(
                available_count=available_count,
                failed_count=failed_count,
                duration_seconds=duration_seconds,
            )
        )
        self._console.show_model_results_table(result_rows)

        return 0

    def _test_models(
        self,
        client: LiteLlmClient,
        models: Sequence[AvailableModel],
    ) -> list[ModelTestResultRow]:
        result_rows: list[ModelTestResultRow] = []

        with self._console.test_models_progress(len(models)) as progress:
            for model in models:
                progress.update_current_model(model.id)
                model_start_time = monotonic()

                try:
                    response = client.ask_model(model_id=model.id, prompt=TEST_PROMPT)
                    duration_seconds = monotonic() - model_start_time
                    result_rows.append(
                        ModelTestResultRow(
                            model_name=model.id,
                            status="ok",
                            duration_seconds=duration_seconds,
                            response=self._format_response(response),
                        )
                    )
                except OpenAIError as error:
                    duration_seconds = monotonic() - model_start_time
                    result_rows.append(
                        ModelTestResultRow(
                            model_name=model.id,
                            status="failed",
                            duration_seconds=duration_seconds,
                            response=self._format_response(self._extract_error_message(error)),
                        )
                    )
                finally:
                    progress.advance()

        return result_rows

    def _format_response(self, response: str) -> str:
        normalized_response = " ".join(response.split())

        if len(normalized_response) <= MAX_RESPONSE_LENGTH:
            return normalized_response

        return f"{normalized_response[:MAX_RESPONSE_LENGTH]}..."

    def _extract_error_message(self, error: OpenAIError) -> str:
        error_message = getattr(error, "message", None)

        if isinstance(error_message, str) and error_message.strip() != "":
            return error_message

        error_text = str(error)
        _, separator, payload_text = error_text.partition(" - ")

        if separator == "":
            return error_text

        try:
            payload = ast.literal_eval(payload_text)
        except (SyntaxError, ValueError):
            return error_text

        message = self._read_nested_error_message(payload)

        if message is None:
            return error_text

        return message

    def _read_nested_error_message(self, payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None

        error_payload = payload.get("error")

        if not isinstance(error_payload, dict):
            return None

        message = error_payload.get("message")

        if not isinstance(message, str):
            return None

        return message
