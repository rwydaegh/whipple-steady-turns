.PHONY: figures test lint

figures:
	uv run python figures/make_all.py

test:
	uv run pytest

lint:
	uvx ruff check .
