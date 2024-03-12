#!/bin/bash
#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if [[ $1 == "nightly" ]]; then
    TAG=$(git log -1 --format="%h")
    BRANCH=$(git branch --show-current)
else
    # get version tag
    TAG="v$VERSION"
fi

set -ex

# build and publish docs
pushd docs
make clean
make html
git worktree add --track -b gh-pages _site origin/gh-pages

pushd _site
if [[ $1 == "nightly" ]]; then
    # draft copy
    api_dest=api/python-draft
else
    # release copy
    api_dest=api/python
    # also copy site wide changes for release
    cp -r ../site/* .
fi

# in _site
mkdir -p $api_dest
cp -r ../build/html/* $api_dest/

git add --all
dff=$(git diff --staged --stat)
repo_url=$(git config --get remote.origin.url)
url=${repo_url#https://}
github_account=${GITHUB_ACCOUNT:-nvauto}
if [[ -n $dff ]]; then
    git commit -m "Update draft api docs to commit ${TAG} on ${BRANCH}"
    git push -f https://${github_account}:${GITHUB_TOKEN}@${url} gh-pages
fi

popd #_site
git worktree remove _site --force
popd
