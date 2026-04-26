"""Command line entrypoint for LiteLLM management."""

import argparse
from collections.abc import Sequence

from litellm_management.features.registry import FeatureRegistry, create_feature_registry


class LitellmManagementCli:
    """Thin CLI orchestrator that delegates work to registered features."""

    def __init__(self, feature_registry: FeatureRegistry) -> None:
        self._feature_registry = feature_registry

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Parse arguments and run the selected feature."""
        parser = self._create_parser()
        parsed_args = parser.parse_args(argv)
        selected_features = self._feature_registry.get_selected_features(parsed_args)

        if not selected_features:
            parser.print_help()
            return 0

        if len(selected_features) > 1:
            parser.error("Choose exactly one feature flag.")

        return selected_features[0].run()

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="litellm_management",
            description="Tools and scripts for managing LiteLLM.",
        )

        for feature in self._feature_registry.features:
            parser.add_argument(
                feature.definition.flag,
                action="store_true",
                help=feature.definition.description,
            )

        return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LiteLLM management CLI."""
    cli = LitellmManagementCli(create_feature_registry())
    return cli.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
