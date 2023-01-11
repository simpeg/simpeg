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
FLAKE8_IGNORE = E121,E123,E126,E226,E24,E704,W503,W504

FLAKE8_IGNORE += B001
FLAKE8_IGNORE += B006
FLAKE8_IGNORE += B007
FLAKE8_IGNORE += B008
FLAKE8_IGNORE += B015
FLAKE8_IGNORE += B017
FLAKE8_IGNORE += B023
FLAKE8_IGNORE += D100
FLAKE8_IGNORE += D101
FLAKE8_IGNORE += D102
FLAKE8_IGNORE += D103
FLAKE8_IGNORE += D104
FLAKE8_IGNORE += D105
FLAKE8_IGNORE += D107
FLAKE8_IGNORE += D200
FLAKE8_IGNORE += D201
FLAKE8_IGNORE += D202
FLAKE8_IGNORE += D205
FLAKE8_IGNORE += D208
FLAKE8_IGNORE += D209
FLAKE8_IGNORE += D210
FLAKE8_IGNORE += D211
FLAKE8_IGNORE += D300
FLAKE8_IGNORE += D400
FLAKE8_IGNORE += D401
FLAKE8_IGNORE += D402
FLAKE8_IGNORE += D403
FLAKE8_IGNORE += D412
FLAKE8_IGNORE += D414
FLAKE8_IGNORE += D419
FLAKE8_IGNORE += E301
FLAKE8_IGNORE += E302
FLAKE8_IGNORE += E306
FLAKE8_IGNORE += E401
FLAKE8_IGNORE += E402
FLAKE8_IGNORE += E711
FLAKE8_IGNORE += E712
FLAKE8_IGNORE += E722
FLAKE8_IGNORE += E731
FLAKE8_IGNORE += F401
FLAKE8_IGNORE += F403
FLAKE8_IGNORE += F405
FLAKE8_IGNORE += F522
FLAKE8_IGNORE += F523
FLAKE8_IGNORE += F524
FLAKE8_IGNORE += F541
FLAKE8_IGNORE += F811
FLAKE8_IGNORE += F821
FLAKE8_IGNORE += F841
FLAKE8_IGNORE += RST201
FLAKE8_IGNORE += RST203
FLAKE8_IGNORE += RST206
FLAKE8_IGNORE += RST210
FLAKE8_IGNORE += RST212
FLAKE8_IGNORE += RST213
FLAKE8_IGNORE += RST215
FLAKE8_IGNORE += RST219
FLAKE8_IGNORE += RST301
FLAKE8_IGNORE += RST303
FLAKE8_IGNORE += RST304
FLAKE8_IGNORE += RST307
FLAKE8_IGNORE += RST499
FLAKE8_IGNORE += W291
FLAKE8_IGNORE += W293

null  :=
space := $(null) #
comma := ,

FLAKE8_IGNORE := $(subst $(space),$(comma),$(strip $(FLAKE8_IGNORE)))

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
