#!/bin/bash
set -x #echo on

conda activate simpeg-test
pytest $TEST_TARGET --cov-config=pyproject.toml -v -W ignore::DeprecationWarning
coverage xml