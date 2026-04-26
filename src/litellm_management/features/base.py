"""Base abstractions for CLI features."""

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

    @abstractmethod
    def run(self) -> int:
        """Run the feature and return a process exit code."""

