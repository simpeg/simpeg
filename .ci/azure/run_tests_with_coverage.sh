#!/bin/bash
set -ex #echo on and exit if any line fails

source activate simpeg-test
pytest $TEST_TARGET --cov --cov-config=pyproject.toml -v -W ignore::DeprecationWarning
coverage xml