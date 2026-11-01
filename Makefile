.PHONY: check

check:
	uv run --locked ruff check --fix .
	uv run --locked ruff format .
	uv run --locked pytest
