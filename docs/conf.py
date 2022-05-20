# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
# sys.path.insert(0, os.path.abspath('tutorials'))
from scar.main.__version__ import __version__, _copyright

# -- Project information -----------------------------------------------------


project = "scAR"
copyright = _copyright
author = "Caibin Sheng"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx_disqus.disqus",
    "sphinxarg.ext",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "autodocsumm",
    "matplotlib.sphinxext.plot_directive",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
autosummary_generate = True
# Add type of source files

# source_suffix = ['.rst', '.md'] #, '.ipynb'
nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["tutorials"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["Thumbs.db", ".DS_Store"]

# Add comments
disqus_shortname = "scar-discussion"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_logo = "img/scAR_logo_transparent.png"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
}
autodoc_mock_imports = ["django"]
autodoc_default_options = {
    "autosummary": True,
}
numpydoc_show_class_members = False

# Options for plot examples
plot_include_source = True
plot_formats = [("png", 120)]
plot_html_show_formats = False
plot_html_show_source_link = False
