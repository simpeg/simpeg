STYLE_CHECK_FILES=SimPEG examples tutorials tests

# Define flake8 warnings that should be catched for now.
# Every time a new warning is solved througout the entire project, it should be
# added to this list.
# This list is only meant to be used in the flake-permissive target and it's
# a temprary solution until every flake8 warning is solved in SimPEG
FLAKE8_SELECT=W605,D301


.PHONY: build coverage lint graphs tests docs check black flake

build:
	python setup.py build_ext --inplace

coverage:
	nosetests --logging-level=INFO --with-coverage --cover-package=SimPEG --cover-html
	open cover/index.html

lint:
	pylint --output-format=html SimPEG > pylint.html

graphs:
	pyreverse -my -A -o pdf -p SimPEG SimPEG/**.py SimPEG/**/**.py

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

flake-permissive:
	flake8 --version
	flake8 ${FLAKE8_OPTS} --select ${FLAKE8_SELECT} ${STYLE_CHECK_FILES}
