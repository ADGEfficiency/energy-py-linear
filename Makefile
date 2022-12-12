.PHONY: all clean
all: test

#  SETUP
.PHONY: setup setup-test setup-static setup-check
setup:
	pip install --upgrade pip -q
	pip install poetry -c ./constraints.txt -q
	poetry install --with main -q
setup-test: setup
	poetry install --with test -q
setup-static: setup
	poetry install --with static -q
setup-check: setup
	poetry install --with check -q

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

#  CHECKS, FORMATTING & LINTING
.PHONY: check check
check: setup-check
	flake8 --extend-ignore E501
	isort --check **/*.py --profile black
	black --check **/*.py
	poetry lock --check
lint: setup-check
	isort **/*.py --profile black
	black **/*.py
	poetry lock --no-update
