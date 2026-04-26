LiteLLM Management tool
=======================

This repository contains tools and scripts for managing LiteLLM tool.

Architecture
------------
- The project is a Python application managed with Poetry.
- Runtime and development commands should be executed through Poetry.
- The package uses the `src/` layout; application code lives in `src/litellm_management/`.
- The primary CLI entrypoint is `poetry run litellm_management`.
- The repository currently does not include a test suite.
- Use object-oriented design when it provides clearer boundaries.
- All Python code must use type annotations.
- Use Pydantic for runtime validation of CLI metadata, configuration, and structured data.
- Keep `.env` local-only; commit `.env.example` for required environment variables.
- Use the official `openai` package for OpenAI-compatible API communication; do not implement custom HTTP calls for it.

Feature implementation rules
----------------------------
- CLI functionality is implemented as `Feature` classes in `src/litellm_management/features/`.
- Each user-facing functionality must be represented by exactly one `Feature` class.
- Each `Feature` class must live in its own file in `src/litellm_management/features/`.
- Each `Feature` must expose one dedicated CLI flag through `FeatureDefinition`.
- Define the flag and help text with `FeatureDefinition(flag=..., description=...)`.
- Implement the feature behavior in `run() -> int`.
- Return process-style exit codes from `run()`; use `0` for success.
- Register every feature in `create_feature_registry()` in `src/litellm_management/features/registry.py`.
- Keep `src/litellm_management/cli.py` thin; it should only parse arguments and delegate to registered features.
- Do not add feature-specific business logic or feature-specific branching to `cli.py`.
- If a feature needs structured input, output, configuration, or metadata, model it with Pydantic.
- Keep OpenAI-compatible API calls outside feature classes in dedicated client/service objects.
- Route rich CLI presentation through dedicated UI objects; avoid direct `print()` calls in feature classes.
- Prefer Open-Closed Principle friendly changes: add new feature classes and supporting objects instead of modifying CLI orchestration.
- Use object-oriented strategies for behavior that may vary between features or providers.

Repository updates rules
------------------------
- When doing update, always think if it is good idea to update the AGENTS.md file.
