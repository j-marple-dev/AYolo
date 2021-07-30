format:
	black .
	isort .
	docformatter -i -r . --wrap-summaries 88 --wrap-descriptions 88

lint:
	env PYTHONPATH=. pytest --pylint --mypy --flake8

test:
	black . --check
	isort . --check

dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install
