.PHONY: all

all: test

#  SETUP
.PHONY: setup setup-test setup-static setup-check setup-docs
QUIET := -q

setup:
	pip install --upgrade pip $(QUIET)
	pip install poetry -c ./constraints.txt $(QUIET)
	poetry install --with main $(QUIET)

setup-test: setup
	poetry install --with test $(QUIET)

setup-static: setup
	poetry install --with static $(QUIET)

setup-check: setup
	poetry install --with check $(QUIET)

#  manage docs dependencies separately because
#  we build docs on netlify
#  netlify only has Python 3.8
#  maybe could change this now we use mike
#  as we don't run a build on netlify anymore
setup-docs:
	pip install -r ./docs/requirements.txt $(QUIET)

#  TEST
.PHONY: test test-docs clean-test-docs test-ci test-validate
DISABLE_LOGGERS = ""
export

test: setup-test clean-test-docs test-docs
	pytest tests --showlocals --full-trace --tb=short -v -x -s --color=yes --testmon --pdb

test-docs: clean-test-docs
	mkdir -p ./tests/phmdoctest
	python -m phmdoctest README.md --outfile tests/phmdoctest/test_readme.py
	python -m phmdoctest ./docs/docs/validation.md --outfile tests/phmdoctest/test_validate.py
	python -m phmdoctest ./docs/docs/how-to/dispatch-forecast.md --outfile tests/phmdoctest/test_forecast.py
	python -m phmdoctest ./docs/docs/how-to/price-carbon.md --outfile tests/phmdoctest/test_carbon.py
	python -m phmdoctest ./docs/docs/how-to/dispatch-assets.md --outfile tests/phmdoctest/test_dispatch_assets.py

clean-test-docs:
	rm -rf ./tests/phmdoctest

test-ci: setup-test clean-test-docs test-docs
	coverage run -m pytest tests --tb=short --show-capture=no
	coverage report -m

#  during debug
#  could just use `test-docs` really
test-validate:
	mkdir -p tests/phmdoctest
	python -m phmdoctest ./docs/docs/validation.md --outfile tests/phmdoctest/test_validate.py
	pytest tests/phmdoctest/test_validate.py --showlocals --full-trace --tb=short -v -x --lf -s --color=yes

#  CHECK
.PHONY: check lint static

check: lint static

#  STATIC TYPING

static: setup-static
	rm -rf ./tests/phmdoctest
	mypy --pretty ./energypylinear
	mypy --pretty ./tests
	mypy --pretty ./examples

#  LINTING

lint: setup-check
	rm -rf ./tests/phmdoctest
	ruff check . --ignore E501 --extend-exclude=__init__.py,poc
	isort --check **/*.py --profile black
	black --check **/*.py
	poetry lock --check

#  FORMATTING

.PHONY: format
format: setup-check
	isort **/*.py --profile black
	black **/*.py

#  PUBLISH TO PYPI

-include .env.secret
.PHONY: publish

publish: setup
	poetry build
	@poetry config pypi-token.pypi $(PYPI_TOKEN)
	poetry publish
	#  TODO publish docs

#  DOCS
.PHONY: docs mike-deploy

docs: setup-docs
	#  `mike serve` will show docs for the different versions
	#  `mkdocs serve` will show docs for the current version in markdown
	#  `mkdocs serve` will usually be more useful during development
	cd docs; mkdocs serve; cd ..

#  this deploys the current docs to the docs branch
#  -u = update aliases of this $(VERSION) to latest
#  -b = branch - aligns with the branch name we build docs off
#  -r = Github remote
#  -p = push
#  TODO - get VERSION from pyproject.toml
#  TODO - this is not used in CI anywhere yet
mike-deploy: setup-docs
	cd docs; mike deploy $(VERSION) latest -u -b mike-pages -r origin -p
