# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spark-rapids-ml'
copyright = '2023, NVIDIA'
author = 'NVIDIA'
release = '23.2.0'

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

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'pyspark': ('https://spark.apache.org/docs/latest/api/python', None),
    'cuml': ('https://docs.rapids.ai/api/cuml/stable', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_static_path = []

def setup(app):
    app.add_css_file("https://docs.rapids.ai/assets/css/custom.css")
    app.add_js_file("https://docs.rapids.ai/assets/js/custom.js", loading_method="defer")
