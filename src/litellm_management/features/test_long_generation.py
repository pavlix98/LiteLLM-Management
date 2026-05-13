"""Feature for testing long LiteLLM Responses API generation."""

import argparse

from openai import OpenAIError
from pydantic import BaseModel, Field

from litellm_management.cli_ui import CliConsole, LongGenerationResultSummary
from litellm_management.config import LiteLlmConfigLoader, MissingLiteLlmApiTokenError
from litellm_management.features.base import Feature, FeatureDefinition
from litellm_management.litellm_client import LiteLlmClient


LONG_GENERATION_MODEL_ID = "GPT-5.5"
LONG_GENERATION_PROMPT = "Napiš esej o historii Česka, minimálně 3000 slov, velmi detailně."
LONG_GENERATION_TIMEOUT_SECONDS = 900.0
MAX_RESPONSE_PREVIEW_LENGTH = 1500


class LongGenerationSettings(BaseModel):
    """Validated settings for the long generation test."""

    model_id: str = Field(default=LONG_GENERATION_MODEL_ID, min_length=1)


class TestLongGenerationFeature(Feature):
    """Test whether long streamed Responses API generation works."""

    def __init__(self, console: CliConsole | None = None) -> None:
        super().__init__(
            FeatureDefinition(
                flag="--test-long-generation",
                description="Test long streamed LiteLLM Responses API generation.",
            )
        )
        self._console = console or CliConsole()
        self._settings = LongGenerationSettings()

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the feature flag and long generation parameters."""
        super().add_arguments(parser)
        parser.add_argument(
            "--long-generation-model",
            default=LONG_GENERATION_MODEL_ID,
            help=f"Model ID to test with long generation. Default: {LONG_GENERATION_MODEL_ID}.",
        )

    def configure(self, parsed_args: argparse.Namespace) -> None:
        """Read long generation settings from parsed CLI arguments."""
        self._settings = LongGenerationSettings(
            model_id=parsed_args.long_generation_model,
        )

    def run(self) -> int:
        """Run the long generation check."""
        try:
            config = LiteLlmConfigLoader().load()
            self._console.show_feature_header(
                feature_name="Test long generation",
                description="Sends a long streamed Responses API prompt and measures TTFB and total time.",
                endpoint_url=config.base_url,
                test_prompt=LONG_GENERATION_PROMPT,
            )

            client = LiteLlmClient(config)
            with self._console.long_generation_progress():
                result = client.generate_long_response(
                    model_id=self._settings.model_id,
                    prompt=LONG_GENERATION_PROMPT,
                    timeout_seconds=LONG_GENERATION_TIMEOUT_SECONDS,
                )
        except MissingLiteLlmApiTokenError as error:
            self._console.show_error(str(error))
            return 1
        except OpenAIError as error:
            self._console.show_error(f"LiteLLM API error: {error}")
            return 1
        except RuntimeError as error:
            self._console.show_error(f"LiteLLM stream error: {error}")
            return 1

        self._console.show_long_generation_result(
            LongGenerationResultSummary(
                model_name=result.model_id,
                ttfb_seconds=result.ttfb_seconds,
                total_seconds=result.total_seconds,
                event_count=result.event_count,
                character_count=result.character_count,
                word_count=result.word_count,
                response_preview=self._format_response_preview(result.output_text),
            )
        )

        return 0

    def _format_response_preview(self, response: str) -> str:
        normalized_response = response.strip()

        if len(normalized_response) <= MAX_RESPONSE_PREVIEW_LENGTH:
            return normalized_response

        return f"{normalized_response[:MAX_RESPONSE_PREVIEW_LENGTH]}..."
