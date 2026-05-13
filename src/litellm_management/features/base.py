"""Base abstractions for CLI features."""

import argparse

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class FeatureDefinition(BaseModel):
    """Validated CLI metadata for a feature."""

    flag: str = Field(pattern=r"^--[a-z][a-z0-9-]*$")
    description: str = Field(min_length=1)


class Feature(ABC):
    """Strategy-style contract for one CLI feature."""

    def __init__(self, definition: FeatureDefinition) -> None:
        self.definition = definition

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add this feature's CLI flag and optional feature parameters."""
        parser.add_argument(
            self.definition.flag,
            action="store_true",
            help=self.definition.description,
        )

    def configure(self, parsed_args: argparse.Namespace) -> None:
        """Read parsed CLI arguments before running the feature."""

    @abstractmethod
    def run(self) -> int:
        """Run the feature and return a process exit code."""
