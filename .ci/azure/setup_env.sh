#!/bin/bash
set -x #echo on

# TF_BUILD is set to True on azure pipelines.
if $TF_BUILD
then
  conda update -n base conda
fi

cp .ci/environment_test.yml environment_test_with_pyversion.yml
echo "  - python="$PYTHON_VERSION >> environment_test_with_pyversion.yml

conda env create --force --file environment_test_with_pyversion.yml
rm environment_test_with_pyversion.yml

conda activate simpeg-test
if $TF_BUILD
then
  pip install pytest-azurepipelines
fi

pip install --no-deps -e .

echo "Conda Environment:"
conda list

echo "Installed SimPEG version:"
python -c "import simpeg; print(simpeg.__version__)"