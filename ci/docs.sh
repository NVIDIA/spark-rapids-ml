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

# get version tag
TAG=$(git describe --tag)
if [[ $? != 0 ]]; then
    echo "Can only deploy from a version tag."
    exit 1
fi

set -ex

# install dependences
pushd python
pip install -r requirements_dev.txt
popd

# build and publish docs
pushd docs
make html
git worktree add --track -b gh-pages _site gh-pages
cp -r build/html/* _site/api/python
cp -r site/* _site
pushd _site
git add --all
git commit -m "${TAG}"
git push origin gh-pages
popd #_site
git worktree remove _site
popd
