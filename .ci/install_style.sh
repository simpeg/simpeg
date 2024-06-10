#!/bin/bash
set -x #echo on

# get directory of this script
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo $script_dir

style_script=$script_dir/parse_style_requirements.py
out_file=$script_dir/requirements_style.txt

# generate the style_requirements file
python $style_script > $out_file

# then install them.
pip install -r $out_file

