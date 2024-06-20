#!/usr/bin/env bash

# Push built docs to gh-pages branch in simpeg-docs repository

set -ex #echo on and exit if any line fails

# Capture simpeg version
version=$(git tag --points-at HEAD)
if [[ -z $version ]]; then
echo "Version could not be obtained from tag. Exiting."
exit 1
fi

# Capture hash of last commit in simpeg
commit=$(git rev-parse --short HEAD)

# Clone the repo where we store the documentation
git clone -q --branch gh-pages --depth 1 "https://${GH_TOKEN}@github.com/simpeg/simpeg-docs.git"
cd simpeg-docs

# Move the built docs to a new dev folder
cp -r "$BUILD_SOURCESDIRECTORY/docs/_build/html" "$version"
cp "$BUILD_SOURCESDIRECTORY/docs/README.md" .

# Add .nojekyll if missing
touch .nojekyll

# Update latest symlink
rm -f latest
ln -s "$version" latest

# Add new docs and relevant files
git add "$version" README.md .nojekyll latest

# List files in working directory and show git status
ls -la
git status

# Commit the new docs.
message="Azure CI deploy ${version} from ${commit}"
echo -e "\nMaking a new commit:"
git commit -m "$message"

# Make the push quiet just in case there is anything that could
# leak sensitive information.
echo -e "\nPushing changes to simpeg/simpeg-docs."
git push -fq origin gh-pages 2>&1 >/dev/null
echo -e "\nFinished uploading generated files."
