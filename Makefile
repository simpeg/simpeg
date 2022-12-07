STYLE_CHECK_FILES=SimPEG examples tutorials tests

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
	flake8 ${STYLE_CHECK_FILES}
