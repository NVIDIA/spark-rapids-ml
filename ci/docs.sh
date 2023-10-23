#!/bin/bash
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
else
    # get version tag
    TAG=$(git describe --tag)
    if [[ $? != 0 ]]; then
        echo "Can only deploy stable release docs from a version tag."
        exit 1
    fi
fi

set -ex

# build and publish docs
pushd docs
make clean
make html
git worktree add --track -b gh-pages _site origin/gh-pages

api_dest=""
pushd _site
if [[ $1 == "nightly" ]]; then
    # set api_dest to trigger copy only if commit has changed since last update
    prev_commit_mesg=$( git log -1 --format="%s" )
    if [[ $prev_commit_mesg != $TAG ]]; then
        api_dest=api/python-draft
    fi
else
    # release copy
    api_dest=api/python
    # also copy site wide changes for release
    cp -r ../site/* .
fi

# in _site
if [[ -n $api_dest ]]; then
    mkdir -p $api_dest
    cp -r ../build/html/* $api_dest/

    git add --all

    git commit -m "${TAG}"
    git push origin gh-pages
fi

popd #_site
git worktree remove _site
popd
