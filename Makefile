STYLE_CHECK_FILES = simpeg examples tutorials tests

.PHONY: help build coverage lint graphs tests docs check black flake

help:
	@echo "Commands:"
	@echo ""
	@echo "  check       run code style and quality checks (black and flake8)"
	@echo "  black       checks code style with black"
	@echo "  flake       checks code style with flake8"
	@echo "  flake-all   checks code style with flake8 (full set of rules)"
	@echo ""

build:
	python setup.py build_ext --inplace

coverage:
	nosetests --logging-level=INFO --with-coverage --cover-package=simpeg --cover-html
	open cover/index.html

lint:
	pylint --output-format=html simpeg> pylint.html

graphs:
	pyreverse -my -A -o pdf -p simpeg simpeg/**.py simpeg/**/**.py

tests:
	nosetests --logging-level=INFO

docs:
	cd docs;make html

clean:
	cd docs;make clean
	find . -name "*.pyc" | xargs -I {} rm -v "{}"

check: black flake

black:
	black --version
	black --check ${STYLE_CHECK_FILES}

flake:
	flake8 --version
	flake8 ${FLAKE8_OPTS} ${STYLE_CHECK_FILES}

flake-all:
	flake8 --version
	flake8 ${FLAKE8_OPTS} --ignore "" ${STYLE_CHECK_FILES}
