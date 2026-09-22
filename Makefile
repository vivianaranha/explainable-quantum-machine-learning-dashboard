.PHONY: install app reference test lint format quality notebook build clean

install:
	python -m pip install -e ".[dev,notebook]"

app:
	streamlit run app.py

reference:
	xqml-dashboard run --config configs/default.json --output examples/reference-run

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

quality: lint test build

notebook:
	python scripts/execute_notebook.py

build:
	python -m build

clean:
	python scripts/clean_generated.py

