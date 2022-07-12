format:
	brunette . --config=setup.cfg
	isort .
lint:
	pytest . --pylint --flake8 --mypy