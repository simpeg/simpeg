#!/bin/bash
set -x -o pipefail #echo on and return non-zero status if any line fails

source activate simpeg-test
pytest $TEST_TARGET --cov --cov-config=pyproject.toml -v -W ignore::DeprecationWarning
coverage xml