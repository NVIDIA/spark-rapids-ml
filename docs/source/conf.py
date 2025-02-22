# Copyright (c) 2024, NVIDIA CORPORATION.
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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spark-rapids-ml'
copyright = '2024, NVIDIA'
author = 'NVIDIA'
release = '25.02.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
]

numpydoc_show_class_members = False

autodoc_inherit_docstrings = False

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'pyspark': ('https://spark.apache.org/docs/latest/api/python', None),
    'cuml': ('https://docs.rapids.ai/api/cuml/stable', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_show_sourcelink = False

import inspect
from spark_rapids_ml.utils import _unsupported_methods_attributes

_unsupported_by_class = {}
def autodoc_skip_member(app, what, name, obj, skip, options):
    # adapted from this https://github.com/sphinx-doc/sphinx/issues/9533#issuecomment-962007846
    doc_class=None
    for frame in inspect.stack():
        if frame.function == "get_members":
            doc_class = frame.frame.f_locals["obj"]
            break
    
    exclude = skip
    if doc_class:
        if doc_class not in _unsupported_by_class:
            _unsupported_by_class[doc_class] = _unsupported_methods_attributes(doc_class)

        exclude = name in _unsupported_by_class[doc_class]

    # return True if (skip or exclude) else None  # Can interfere with subsequent skip functions.
    return True if exclude or skip else None

def setup(app):
    app.add_css_file("https://docs.rapids.ai/assets/css/custom.css")
    app.add_js_file("https://docs.rapids.ai/assets/js/custom.js", loading_method="defer")
    app.connect('autodoc-skip-member', autodoc_skip_member)
