PROJECT = simpeg
STYLE_CHECK_FILES = simpeg examples tutorials tests
PYTEST_OPTIONS = -v --cov-config=.coveragerc --cov=${PROJECT} --cov-report=xml --cov-report=html -W ignore::DeprecationWarning
PYTEST_TARGET = ${PROJECT}

.PHONY: help docs install test check black flake flake-all

help:
	@echo "Commands:"
	@echo ""
	@echo "  install     install simpeg in edit (a.k.a developer) mode"
	@echo "  test        run full test suite"
	@echo "  check       run code style and quality checks (black and flake8)"
	@echo "  black       checks code style with black"
	@echo "  flake       checks code style with flake8"
	@echo "  flake-all   checks code style with flake8 (full set of rules)"
	@echo ""

install:
	python -m pip install --no-deps -e .

test:
	pytest ${PYTEST_OPTIONS} ${PYTEST_TARGET}

docs:
	make -C docs html

clean:
	make -C docs clean
	find . -name "*.pyc" | xargs -I {} rm -v "{}"

check: black flake

black:
	black --version
	black --check ${STYLE_CHECK_FILES}

flake:
	flake8 --version
	flake8 ${STYLE_CHECK_FILES}

flake-all:
	flake8 --version
	flake8 --ignore "" ${STYLE_CHECK_FILES}
