.PHONY: all clean

all: test

clean:
	rm -rf .pytest_cache .hypothesis .mypy_cache .ruff_cache __pycache__ .coverage logs .coverage*


#  ----- SETUP -----
#  installation of dependencies

.PHONY: setup-pip-poetry setup-test setup-static setup-check setup-docs
QUIET := -q
PIP_CMD=pip

setup-pip-poetry:
	$(PIP_CMD) install --upgrade pip $(QUIET)
	$(PIP_CMD) install poetry==1.7.0 $(QUIET)

setup: setup-pip-poetry
	poetry install --with main $(QUIET)

setup-test: setup-pip-poetry
	poetry install --with test $(QUIET)

setup-static: setup-pip-poetry
	poetry install --with static $(QUIET)

setup-check: setup-pip-poetry
	poetry install --with check $(QUIET)

#  manage docs dependencies separately because
#  we build docs on netlify
#  netlify only has Python 3.8
#  TODO could change this now we use mike
#  as we don't run a build on netlify anymore
setup-docs:
	$(PIP_CMD) install -r ./docs/requirements.txt $(QUIET)


#  ----- TEST -----
#  documentation tests and unit tests

.PHONY: test generate-test-docs test-docs
PARALLEL = auto
TEST_ARGS =
export

test: setup-test test-docs
	pytest tests --cov=energypylinear --cov-report=html --cov-report=term-missing -n $(PARALLEL) --color=yes --durations=5 --verbose --ignore tests/phmdoctest $(TEST_ARGS)
	# -coverage combine
	# -coverage html
	-coverage report
	python tests/assert-test-coverage.py $(TEST_ARGS)

generate-test-docs: setup-test
	bash ./tests/generate-test-docs.sh

test-docs: setup-test generate-test-docs
	pytest tests/phmdoctest -n 1 --dist loadfile --color=yes --verbose $(TEST_ARGS)


#  ----- CHECK -----
#  linting and static typing

.PHONY: check lint static

check: lint static

MYPY_ARGS=--pretty
static: setup-static
	rm -rf ./tests/phmdoctest
	mypy --version
	mypy $(MYPY_ARGS) ./energypylinear
	mypy $(MYPY_ARGS) ./tests --explicit-package-bases

lint: setup-check
	rm -rf ./tests/phmdoctest
	flake8 --extend-ignore E501,DAR --exclude=__init__.py,poc
	ruff check . --ignore E501 --extend-exclude=__init__.py,poc
	isort --check **/*.py --profile black
	ruff format --check **/*.py
	poetry check

CHECK_DOCSTRINGS=./energypylinear/objectives.py ./energypylinear/assets/battery.py ./energypylinear/assets/renewable_generator.py

# currently only run manually
lint-docstrings:
	flake8 --extend-ignore E501 --exclude=__init__.py,poc --exit-zero $(CHECK_DOCSTRINGS)
	pydocstyle $(CHECK_DOCSTRINGS)
	# pylint $(CHECK_DOCSTRINGS)


#  ----- FORMATTING -----
#  formatting code

.PHONY: format
format: setup-check
	isort **/*.py --profile black
	ruff format **/*.py


#  ----- PUBLISH ------
#  updating package on pypi

.PHONY: publish
-include .env.secret

publish: setup
	poetry build
	@poetry config pypi-token.pypi $(PYPI_TOKEN)
	poetry publish
	#  TODO publish docs automatically


#  ----- DOCS ------
#  mkdocs documentation

.PHONY: docs mike-deploy

generate-docs-images: setup
	python ./docs/generate-plots.py

docs: setup-docs
	#  `mike serve` will show docs for the different versions
	#  `mkdocs serve` will show docs for the current version in markdown
	#  `mkdocs serve` will usually be more useful during development
	cd docs; mkdocs serve -a localhost:8004; cd ..

#  TODO currently run manually - should be automated with publishing
#  this deploys the current docs to the docs branch
#  -u = update aliases of this $(VERSION) to latest
#  -b = branch - aligns with the branch name we build docs off
#  -r = Github remote
#  -p = push
#  TODO - get VERSION from pyproject.toml
#  TODO - this is not used in CI anywhere yet
mike-deploy: setup-docs generate-docs-images
	cd docs; mike deploy $(VERSION) latest -u -b mike-pages -r origin -p
