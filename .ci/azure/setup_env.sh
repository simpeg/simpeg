#!/bin/bash
set -ex #echo on and exit if any line fails

# TF_BUILD is set to True on azure pipelines.
is_azure=$(echo "${TF_BUILD:-false}" | tr '[:upper:]' '[:lower:]')

if ${is_azure}
then
  # Add conda-forge as channel
  conda config --add channels conda-forge
  # Remove defaults channels
  # (both from ~/.condarc and from /usr/share/miniconda/.condarc)
  conda config --remove channels defaults
  conda config --remove channels defaults --system
  # Update conda
  conda update --yes -n base conda
fi

cp .ci/environment_test.yml environment_test_with_pyversion.yml
echo "  - python="$PYTHON_VERSION >> environment_test_with_pyversion.yml

conda env create --file environment_test_with_pyversion.yml
rm environment_test_with_pyversion.yml


if ${is_azure}
then
  source activate simpeg-test
  pip install pytest-azurepipelines
else
  conda activate simpeg-test
fi

pip install --no-deps --editable .

echo "Conda Environment:"
conda list

echo "Installed SimPEG version:"
python -c "import simpeg; print(simpeg.__version__)"
