#!/bin/bash
set -x #echo on

# get directory of this script
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
style_script=$script_dir/parse_style_requirements.py

# parse the style requirements
requirements=$(python $style_script)

pip install $requirements

