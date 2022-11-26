#  SETUP
.PHONY: setup setup-test
setup:
	poetry install --with main
setup-test:
	poetry install --with main,test

#  TEST
.PHONY: test test-ci
test: setup-test
	pytest tests --showlocals --full-trace --tb=short --show-capture=no -v -x --lf
test-ci: setup-test
	coverage run -m pytest tests --tb=short --show-capture=no
	coverage report -m

#  STATIC TYPING
.PHONY: static
static: setup-checks
	mypy **/*.py --config-file ./mypy.ini --pretty

#  FORMATTING & LINTING
.PHONY: format lint
format: setup-format
	isort **/*.py --profile black
	black **/*.py
lint: setup-format
	isort --check **/*.py --profile black
	black --check **/*.py
	poetry lock --check
