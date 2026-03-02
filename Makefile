.PHONY: install dev serve test lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

serve:
	moltwrath serve --reload

test:
	pytest tests/ -v

lint:
	ruff check moltwrath/ --fix
	ruff format moltwrath/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info dist build .pytest_cache

docker:
	docker compose up --build
