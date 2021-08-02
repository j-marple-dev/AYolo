format:
	black .
	isort .
	docformatter -i -r . --wrap-summaries 88 --wrap-descriptions 88

lint:
	env PYTHONPATH=. pytest --pylint --mypy --flake8 --ignore ./tensorrt_run/youtube_test.py

test:
	black . --check
	isort . --check

dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	# install nvidia dali
	pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda102
	# pre-commit install
