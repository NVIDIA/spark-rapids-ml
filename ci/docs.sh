#!/bin/bash

# get version tag
TAG=$(git describe --tag)
if [[ $? != 0 ]]; then
    echo "Can only deploy from a version tag."
    exit 1
fi

# exit on first fail
set -e

# build and publish docs
pushd docs
make html
git worktree add --track -b gh-pages _site origin/gh-pages
cp -r build/html/* _site/api/python
cp -r site/* _site
pushd _site
git add --all
git commit -m "${TAG}"
git push origin gh-pages
popd #_site
git worktree remove _site
popd
