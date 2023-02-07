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
	rm -f ./tests/test_readme.py
	python -m phmdoctest README.md --outfile tests/test_readme.py
	pytest tests --showlocals --full-trace --tb=short -v -x --lf -s --color=yes --testmon
test-ci: setup-test
	coverage run -m pytest tests --tb=short --show-capture=no
	coverage report -m

#  STATIC TYPING
.PHONY: static
static: setup-static
	rm -rf ./tests/test_readme.py
	mypy --config-file ./mypy.ini --pretty ./energypylinear
	mypy --config-file ./mypy.ini --pretty ./tests

#  LINTING
.PHONY: lint
lint: setup-check
	flake8 --extend-ignore E501
	isort --check **/*.py --profile black
	black --check **/*.py
	poetry lock --check

#  FORMATTING
.PHONY: format
format: setup-check
	isort **/*.py --profile black
	black **/*.py
	poetry lock --no-update

#  CHECK
.PHONY: check
check: lint static

#  PUBLISH
-include .env.secret
.PHONY: publish
publish: setup
	poetry build
	@poetry config pypi-token.pypi $(PYPI_TOKEN)
	poetry publish
