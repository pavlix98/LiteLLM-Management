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

Repository updates rules
------------------------
- When doing update, always think if it is good idea to update the AGENTS.md file.
