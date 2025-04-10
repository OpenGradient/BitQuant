# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands
- Setup: `make venv && source venv/bin/activate && make install`
- Run: `make run` (uses Python 3.13)
- Test all: `make test`
- Test single: `python3.13 -m unittest path/to/test_file.py`
- Lint: `make check` (runs black and mypy)
- Format: `make format` (uses ruff)

## Code Style Guidelines
- Follow PEP 8 naming: `snake_case` for variables/functions, `PascalCase` for classes
- Use absolute imports, organized by standard lib > third-party > project
- Type annotations required for function parameters and return values
- Line length: 88 characters (black default)
- Document classes and functions with docstrings
- Explicit error handling with try/except and meaningful error messages
- Use Pydantic models for data validation and structure
- Prefer returning error dicts over raising exceptions
- Project requires Python 3.8+ (runs with 3.13)