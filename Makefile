STYLE_CHECK_FILES = simpeg examples tutorials tests
GITHUB_ACTIONS=.github/workflows

.PHONY: help docs clean check black flake flake-all check-actions

help:
	@echo "Commands:"
	@echo ""
	@echo "  check         run code style and quality checks (black and flake8)"
	@echo "  black         checks code style with black"
	@echo "  flake         checks code style with flake8"
	@echo "  flake-all     checks code style with flake8 (full set of rules)"
	@echo "  check-actions lint GitHub Actions workflows (with zizmor)"
	@echo ""

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

check-actions:
	zizmor ${GITHUB_ACTIONS}
