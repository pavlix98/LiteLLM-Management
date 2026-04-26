# LiteLLM Management

Tools and scripts for managing LiteLLM.

## Development

```bash
poetry install
cp .env.example .env
poetry run litellm_management
poetry run litellm_management --help
poetry run litellm_management --test-available-models
```

Set `LITELLM_API_TOKEN` in the local `.env` file before running features that call LiteLLM. The `.env` file is local-only and must not be committed.

## Feature Architecture

CLI functionality is implemented as `Feature` classes. The main CLI only parses arguments and delegates execution to the selected feature.

To add a new feature:

1. Create a new file in `src/litellm_management/features/`.
2. Implement a class that extends `Feature`.
3. Define the feature flag and description with `FeatureDefinition`.
4. Add the feature instance to `create_feature_registry()` in `features/registry.py`.
5. Keep business logic inside the feature or objects used by that feature, not in `cli.py`.

Use object-oriented design when it provides clearer boundaries. All Python code must use type annotations. Use Pydantic models for runtime validation of CLI metadata, configuration, and structured data.
