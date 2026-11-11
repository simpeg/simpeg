#!/bin/bash
# Install Miniforge and put its conda on the PATH for the rest of the job.
# The Azure ubuntu images stopped shipping Miniconda with ubuntu-26.04.
set -ex

version="26.9.4-1"
installer="Miniforge3-${version}-Linux-x86_64.sh"
sha256="571b410272cc434f28bc6c5c49ca8ffd4a4d1bcba9f57f1b6baccdb46446cfcc"
prefix="$HOME/miniforge3"

curl -fsSL -o "$installer" \
  "https://github.com/conda-forge/miniforge/releases/download/${version}/${installer}"
echo "${sha256}  ${installer}" | sha256sum --check
bash "$installer" -b -p "$prefix"
rm "$installer"

echo "##vso[task.prependpath]${prefix}/bin"
