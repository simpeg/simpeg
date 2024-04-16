#!/bin/bash
set -eu
set -o pipefail

echo "Installing reviewdog..."
wget -O - -q https://raw.githubusercontent.com/reviewdog/reviewdog/master/install.sh | sh -s -- -b /tmp "${REVIEWDOG_VERSION}"
echo "Reviewdog version: ${REVIEWDOG_VERSION}"

echo "Black version: $(black --version)"

black_exit_val="0"
reviewdog_exit_val="0"

echo "Checking python code"

black_check_output = "$(black --diff --quiet --check .)" || black_exit_val="$?"

echo "${black_check_output}" | /tmp/reviewdog -f="diff" \
  -f.diff.strip=0 \
  -name="black-format" \
  -reporter="github-pr-review" \
  -filter-mode="diff_context" \
  -level="error"\
  -fail_on_error="true" || reviewdog_exit_val="$?"

# Throw error if an error occurred and fail_on_error is true.
if [[ ("${black_exit_val}" -ne '0' || "${reviewdog_exit_val}" -eq "1") ]]; then
  if [[ "${black_exit_val}" -eq "123" ]]; then
    echo "ERROR: Black found a syntax error when checking the" \
      "files (error code: ${black_exit_val})."
    if [[ "${reviewdog_exit_val}" -eq '1' ]]; then
      exit 1
    fi
  else
    exit 1
  fi
fi