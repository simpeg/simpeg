#!/bin/bash
# Push built docs to gh-pages branch in simpeg-docs repository

# Capture version
# TODO: we should be able to get the version from the
# build.sourceBranch variable
version=$(git tag --points-at HEAD)
if [ -n "$version" ]; then
echo "Version could not be obtained from tag. Exiting."
exit 1
fi
# Capture hash of last commit in simpeg
commit=$(git rev-parse --short HEAD)
# Clone the repo where we store the documentation
git clone -q --branch gh-pages --depth 1 https://${GH_TOKEN}@github.com/simpeg/simpeg-docs.git
cd simpeg-docs
# Move the built docs to a new dev folder
cp -r $BUILD_SOURCESDIRECTORY/docs/_build/html "$version"
cp $BUILD_SOURCESDIRECTORY/docs/README.md .
# Add .nojekyll if missing
touch .nojekyll
# Update latest symlink
rm -f latest
ln -s "$version" latest
# Commit the new docs.
git add "$version" README.md .nojekyll latest
message="Azure CI deploy ${version} from ${commit}"
echo -e "\nMaking a new commit:"
git commit -m "$message"
# Make the push quiet just in case there is anything that could
# leak sensitive information.
echo -e "\nPushing changes to simpeg/simpeg-docs."
git push -fq origin gh-pages 2>&1 >/dev/null
echo -e "\nFinished uploading generated files."
