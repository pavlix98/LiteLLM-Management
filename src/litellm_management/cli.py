"""Command line entrypoint for LiteLLM management."""

import argparse
from collections.abc import Sequence

from litellm_management.cli_ui import CliConsole
from litellm_management.features.registry import FeatureRegistry, create_feature_registry


INTERRUPTED_EXIT_CODE = 130


class LitellmManagementCli:
    """Thin CLI orchestrator that delegates work to registered features."""

    def __init__(
        self,
        feature_registry: FeatureRegistry,
        console: CliConsole | None = None,
    ) -> None:
        self._feature_registry = feature_registry
        self._console = console or CliConsole()

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

        try:
            selected_feature = selected_features[0]
            selected_feature.configure(parsed_args)
            return selected_feature.run()
        except KeyboardInterrupt:
            self._console.show_interrupted()
            return INTERRUPTED_EXIT_CODE

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="litellm_management",
            description="Tools and scripts for managing LiteLLM.",
        )

        for feature in self._feature_registry.features:
            feature.add_arguments(parser)

        return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LiteLLM management CLI."""
    cli = LitellmManagementCli(create_feature_registry())
    return cli.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
