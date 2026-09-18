#!/bin/bash
set -x #echo on

source activate simpeg-test
make TEST_TARGET=${TEST_TARGET} test
pytest_retval=$?
coverage xml
exit $pytest_retval
