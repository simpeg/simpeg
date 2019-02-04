#!/bin/bash
#
# Package and upload to PyPI using twine. The env variables TWINE_USERNAME and
# TWINE_PASSWORD must exist with your pypi.org credentials.

# make tar files of the current docs
cd docs
tar -cvzf _build.tar.gz _build

cd content
tar -cvzf examples.tar.gz examples

cd ../../

# upload tar files of the current build to google cloud storage
gsutil cp docs/_build.tar.gz gs://simpeg

gsutil cp docs/content/examples.tar.gz gs://simpeg

gcloud auth activate-service-account --key-file credentials/client-secret.json

export GAE_PROJECT=$GAE_PROJECT


# deploy the docs to google app engine
gcloud config set project $GAE_PROJECT;
gcloud app deploy ./docs/app.yaml --version ${TRAVIS_COMMIT} --promote;
