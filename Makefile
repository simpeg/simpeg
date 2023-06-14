STYLE_CHECK_FILES = SimPEG examples tutorials tests

# Define flake8 warnings that shouldn't be catched for now.
# Every time a new warning is solved througout the entire project, it should be
# removed to this list.
# This list is only meant to be used in the flake-permissive target and it's
# a temprary solution until every flake8 warning is solved in SimPEG.
# The first set of rules (up to W504) are the default ignored ones by flake8.
# Since we are using the --ignore option we are overriding them. They are
# included in the list so we keep ignoring them while running the
# flake-permissive target.
FLAKE8_IGNORE = "E121,E123,E126,E226,E24,E704,W503,W504,\
	B017,\
	B028,\
	D100,\
	D101,\
	D102,\
	D103,\
	D104,\
	D105,\
	D107,\
	D200,\
	D201,\
	D202,\
	D205,\
	D208,\
	D209,\
	D210,\
	D211,\
	D300,\
	D400,\
	D401,\
	D402,\
	D403,\
	D412,\
	D414,\
	D419,\
	E402,\
	E711,\
	E731,\
	F403,\
	F405,\
	F522,\
	F523,\
	F524,\
	F541,\
	F811,\
	F821,\
	RST201,\
	RST203,\
	RST206,\
	RST210,\
	RST212,\
	RST213,\
	RST215,\
	RST219,\
	RST301,\
	RST303,\
	RST304,\
	RST307,\
	RST499,\
	W291,\
	W293,\
"

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
	flake8 ${FLAKE8_OPTS} --ignore ${FLAKE8_IGNORE} ${STYLE_CHECK_FILES}
