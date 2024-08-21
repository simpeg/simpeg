#!/usr/bin/env bash

# Push built docs to dev branch in simpeg-docs repository

set -ex #echo on and exit if any line fails

# ---------------------------
# Push new docs to dev branch
# ---------------------------
# Capture hash of last commit in simpeg
commit=$(git rev-parse --short HEAD)

# Clone the repo where we store the documentation (dev branch)
git clone -q --branch dev --depth 1 "https://${GH_TOKEN}@github.com/simpeg/simpeg-docs.git"
cd simpeg-docs

# Remove all files (but .git folder)
find . -not -path "./.git/*" -not -path "./.git" -delete

# Copy the built docs to the root of the repo
cp -r "$BUILD_SOURCESDIRECTORY"/docs/_build/html/. -t .

# Add new files
git add .

# List files in working directory and show git status
ls -la
git status

# Commit the new docs. Amend to avoid having a very large history.
message="Azure CI deploy dev from ${commit}"
echo -e "\nAmending last commit:"
git commit --amend --reset-author -m "$message"

# Make the push quiet just in case there is anything that could
# leak sensitive information.
echo -e "\nPushing changes to simpeg/simpeg-docs (dev branch)."
git push -fq origin dev 2>&1 >/dev/null
echo -e "\nFinished uploading doc files."

# ----------------
# Update submodule
# ----------------
# Need to fetch the gh-pages branch first (because we clone with shallow depth)
git fetch --depth 1 origin gh-pages:gh-pages

# Switch to the gh-pages branch
git switch gh-pages

# Update the dev submodule
git submodule update --init --recursive --remote dev

# Add updated submodule
git add dev

# List files in working directory and show git status
ls -la
git status

# Commit changes
message="Azure CI update dev submodule from ${commit}"
echo -e "\nMaking a new commit:"
git commit -m "$message"

# Make the push quiet just in case there is anything that could
# leak sensitive information.
echo -e "\nPushing changes to simpeg/simpeg-docs (gh-pages branch)."
git push -q origin gh-pages 2>&1 >/dev/null
echo -e "\nFinished updating submodule dev."
