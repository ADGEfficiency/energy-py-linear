.PHONY: all clean
all: test

#  SETUP
.PHONY: setup setup-test setup-static setup-format
setup:
	python -m pip install --upgrade pip -q
	python -m pip install poetry -c ./constraints.txt -q
	poetry install --with main
setup-test: setup
	poetry install --with test
setup-static: setup
	poetry install --with static
setup-format: setup
	poetry install --with format

#  TEST
.PHONY: test test-ci
test: setup-test
	pytest tests --showlocals --full-trace --tb=short --show-capture=no -v -x --lf
test-ci: setup-test
	coverage run -m pytest tests --tb=short --show-capture=no
	coverage report -m

#  STATIC TYPING
.PHONY: static
static: setup-static
	mypy **/*.py --config-file ./mypy.ini --pretty

#  FORMATTING & LINTING
.PHONY: format lint
format: setup-format
	isort **/*.py --profile black
	black **/*.py
	poetry lock --no-update
lint: setup-format
	isort --check **/*.py --profile black
	black --check **/*.py
	poetry lock --check
