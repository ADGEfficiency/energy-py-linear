.PHONY: all clean

all: test

clean:
	rm -rf .pytest_cache .hypothesis .mypy_cache .ruff_cache __pycache__ .coverage logs .coverage*


#  ----- SETUP -----

.PHONY: setup-pip-poetry setup-test setup-static setup-check setup-docs
QUIET := -q

setup-pip-poetry:
	pip install --upgrade pip $(QUIET)
	pip install poetry==1.7.0 $(QUIET)

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
	pip install -r ./docs/requirements.txt $(QUIET)


#  ----- TEST -----

.PHONY: test test-ci test-docs clean-test-docs test-validate create-test-docs
PARALLEL = auto
TEST_ARGS =
export

test: setup-test test-docs
	pytest tests --cov=energypylinear --cov-report=html -n $(PARALLEL) --color=yes --durations=5 --verbose --ignore tests/phmdoctest $(TEST_ARGS)
	python tests/assert-test-coverage.py $(TEST_ARGS)
	-coverage combine

create-test-docs: setup-test clean-test-docs
	mkdir -p ./tests/phmdoctest
	python -m phmdoctest README.md --outfile tests/phmdoctest/test_readme.py
	python -m phmdoctest ./docs/docs/how-to/custom-objectives.md  --outfile tests/phmdoctest/test_custom_objectives.py
	python -m phmdoctest ./docs/docs/validation/battery.md --outfile tests/phmdoctest/test_validate_battery.py
	python -m phmdoctest ./docs/docs/validation/evs.md --outfile tests/phmdoctest/test_validate_evs.py
	python -m phmdoctest ./docs/docs/validation/heat-pump.md --outfile tests/phmdoctest/test_validate_heat-pump.py
	python -m phmdoctest ./docs/docs/validation/renewable-generator.md --outfile tests/phmdoctest/test_validate_renewable_generator.py
	python -m phmdoctest ./docs/docs/how-to/dispatch-forecast.md --outfile tests/phmdoctest/test_forecast.py
	python -m phmdoctest ./docs/docs/how-to/price-carbon.md --outfile tests/phmdoctest/test_carbon.py
	python -m phmdoctest ./docs/docs/how-to/dispatch-site.md --outfile tests/phmdoctest/test_dispatch_site.py
	python -m phmdoctest ./docs/docs/assets/chp.md --outfile tests/phmdoctest/test_optimize_chp.py
	python -m phmdoctest ./docs/docs/assets/battery.md --outfile tests/phmdoctest/test_optimize_battery.py
	python -m phmdoctest ./docs/docs/assets/evs.md --outfile tests/phmdoctest/test_optimize_evs.py
	python -m phmdoctest ./docs/docs/assets/heat-pump.md --outfile tests/phmdoctest/test_optimize_heat_pump.py
	python -m phmdoctest ./docs/docs/assets/chp.md --outfile tests/phmdoctest/test_optimize_chp.py
	python -m phmdoctest ./docs/docs/assets/renewable-generator.md --outfile tests/phmdoctest/test_optimize_renewable_generator.py

clean-test-docs:
	rm -rf ./tests/phmdoctest

test-docs: clean-test-docs create-test-docs
	pytest tests/phmdoctest -n $(PARALLEL) --dist loadfile --color=yes --verbose $(TEST_ARGS)


#  ----- CHECK -----
#  linting and static typing

.PHONY: check lint static

check: lint static

MYPY_ARGS=--pretty
static: setup-static
	rm -rf ./tests/phmdoctest
	mypy --version
	mypy $(MYPY_ARGS) ./energypylinear
	mypy $(MYPY_ARGS) ./tests

lint: setup-check
	rm -rf ./tests/phmdoctest
	flake8 --extend-ignore E501,DAR --exclude=__init__.py,poc
	ruff check . --ignore E501 --extend-exclude=__init__.py,poc
	isort --check **/*.py --profile black
	ruff format --check **/*.py
	poetry check


#  ----- FORMATTING -----

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
