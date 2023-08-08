.PHONY: all clean

all: test

clean:
	rm -rf .pytest_cache .hypothesis .mypy_cache .ruff_cache __pycache__ .coverage logs .coverage*

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
.PHONY: test test-ci test-docs clean-test-docs test-validate
PARALLEL = auto
ENABLE_FILE_LOGGING = 0
export

test: setup-test clean-test-docs test-docs
	pytest tests/phmdoctest -n $(PARALLEL) --dist loadfile --color=yes --verbose
	pytest tests --cov=energypylinear --cov-report=html -n $(PARALLEL) --color=yes --durations=5 --verbose --ignore tests/phmdoctest

test-ci: test

test-docs: setup-test clean-test-docs
	mkdir -p ./tests/phmdoctest
	python -m phmdoctest README.md --outfile tests/phmdoctest/test_readme.py
	python -m phmdoctest ./docs/docs/validation/battery.md --outfile tests/phmdoctest/test_validate_battery.py
	python -m phmdoctest ./docs/docs/validation/evs.md --outfile tests/phmdoctest/test_validate_evs.py
	python -m phmdoctest ./docs/docs/how-to/dispatch-forecast.md --outfile tests/phmdoctest/test_forecast.py
	python -m phmdoctest ./docs/docs/how-to/price-carbon.md --outfile tests/phmdoctest/test_carbon.py
	python -m phmdoctest ./docs/docs/how-to/dispatch-assets.md --outfile tests/phmdoctest/test_dispatch_assets.py

clean-test-docs:
	rm -rf ./tests/phmdoctest

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
.PHONY: publish
-include .env.secret

publish: setup
	poetry build
	@poetry config pypi-token.pypi $(PYPI_TOKEN)
	poetry publish
	#  TODO publish docs

#  DOCS
.PHONY: docs mike-deploy

generate-docs-images: setup
	python ./docs/generate-plots.py

docs: setup-docs
	#  `mike serve` will show docs for the different versions
	#  `mkdocs serve` will show docs for the current version in markdown
	#  `mkdocs serve` will usually be more useful during development
	cd docs; mkdocs serve -a localhost:8004; cd ..

#  this deploys the current docs to the docs branch
#  -u = update aliases of this $(VERSION) to latest
#  -b = branch - aligns with the branch name we build docs off
#  -r = Github remote
#  -p = push
#  TODO - get VERSION from pyproject.toml
#  TODO - this is not used in CI anywhere yet
mike-deploy: setup-docs
	cd docs; mike deploy $(VERSION) latest -u -b mike-pages -r origin -p
