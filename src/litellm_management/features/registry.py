"""Registry for available CLI features."""

import argparse
from collections.abc import Iterable

from litellm_management.features.base import Feature
from litellm_management.features.test_available_models import TestAvailableModelsFeature


class FeatureRegistry:
    """Holds all feature strategies available to the CLI."""

    def __init__(self, features: Iterable[Feature]) -> None:
        self.features = tuple(features)
        self._validate_unique_flags()

    def get_selected_features(self, parsed_args: argparse.Namespace) -> list[Feature]:
        """Return features selected by parsed CLI flags."""
        selected_features: list[Feature] = []

        for feature in self.features:
            argument_name = feature.definition.flag.removeprefix("--").replace("-", "_")
            if getattr(parsed_args, argument_name):
                selected_features.append(feature)

        return selected_features

    def _validate_unique_flags(self) -> None:
        flags = [feature.definition.flag for feature in self.features]

        if len(flags) != len(set(flags)):
            raise ValueError("Feature flags must be unique.")


def create_feature_registry() -> FeatureRegistry:
    """Create the default feature registry used by the CLI."""
    return FeatureRegistry(
        features=[
            TestAvailableModelsFeature(),
        ]
    )

