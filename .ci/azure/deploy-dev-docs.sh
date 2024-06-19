#!/usr/bin/env bash

# Push built docs to dev branch in simpeg-docs repository

set -ex #echo on and exit if any line fails

# Configure dotglob (needed for rm with glob)
shopt -s dotglob  # configure bash to include dotfiles in * globs
export GLOBIGNORE=".git"  # ignore .git directory in glob

# ---------------------------
# Push new docs to dev branch
# ---------------------------
# Capture hash of last commit in simpeg
commit=$(git rev-parse --short HEAD)

# Clone the repo where we store the documentation (dev branch)
git clone -q --branch dev --depth 1 "https://${GH_TOKEN}@github.com/simpeg/simpeg-docs.git"
cd simpeg-docs

# Remove all files
git rm -rf ./* # remove all files

# Copy the built docs to the root of the repo
cp -r "$BUILD_SOURCESDIRECTORY/docs/_build/html/*" -t .

# Commit the new docs. Amend to avoid having a very large history.
git add .
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

# Commit changes
git add dev
message="Azure CI update dev submodule from ${commit}"
echo -e "\nMaking a new commit:"
git commit -m "$message"

# Make the push quiet just in case there is anything that could
# leak sensitive information.
echo -e "\nPushing changes to simpeg/simpeg-docs (gh-pages branch)."
git push -q origin gh-pages 2>&1 >/dev/null
echo -e "\nFinished updating submodule dev."

# -------------
# Unset dotglob
# -------------
shopt -u dotglob
export GLOBIGNORE=""
