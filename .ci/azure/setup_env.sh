#!/bin/bash
set -x #echo on

# TF_BUILD is set to True on azure pipelines.
is_azure=${TF_BUILD:-false}

if ${is_azure}
then
  conda update --yes -n base conda
fi

cp .ci/environment_test.yml environment_test_with_pyversion.yml
echo "  - python="$PYTHON_VERSION >> environment_test_with_pyversion.yml

conda env create --file environment_test_with_pyversion.yml
rm environment_test_with_pyversion.yml

source activate simpeg-test
if ${is_azure}
then
  pip install pytest-azurepipelines
fi

pip install --no-deps -e .

echo "Conda Environment:"
conda list

echo "Installed SimPEG version:"
python -c "import simpeg; print(simpeg.__version__)"