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
	python -m phmdoctest README.md --outfile tests/test_readme.py
	pytest tests --showlocals --full-trace --tb=short -v -x --lf -s --color=yes
test-ci: setup-test
	coverage run -m pytest tests --tb=short --show-capture=no
	coverage report -m

#  STATIC TYPING
.PHONY: static
static: setup-static
	rm -rf ./tests/test_readme.py
	mypy **/*.py --config-file ./mypy.ini --pretty

#  CHECKS & FORMATTING
.PHONY: check check
check: setup-check
	flake8 --extend-ignore E501
	isort --check **/*.py --profile black
	black --check **/*.py
	poetry lock --check
format: setup-check
	isort **/*.py --profile black
	black **/*.py
	poetry lock --no-update
