#!/bin/bash
set -x #echo on

source activate simpeg-test
pytest $TEST_TARGET --cov --cov-config=pyproject.toml -v -W ignore::DeprecationWarning
pytest_retval=$?
coverage xml
exit $pytest_retval