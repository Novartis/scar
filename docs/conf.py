# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
from scar.main.__version__ import __version__, _copyright

# -- Project information -----------------------------------------------------
project = "scAR"
copyright = _copyright
author = "Caibin Sheng"
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_disqus.disqus",
    "sphinxarg.ext",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "autodocsumm",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
    "sphinx_tabs.tabs",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
autosummary_generate = True
# Add type of source files

nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}

# Add any paths that contain templates here, relative to this directory.

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
html_theme = "pydata_sphinx_theme"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["scar-styles.css"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_logo = "_static/scAR_logo_transparent.png"
html_theme_options = {
    "logo": {
        "image_light": "scAR_logo_white.png",
        "image_dark": "scAR_logo_black.png",
    },
    "pygment_light_style": "tango",
    "pygment_dark_style": "native",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Novartis/scar",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": False,
    "favicons": [
        {
            "rel": "icon",
            "sizes": "32x32",
            "href": "_static/scAR_favicon.png",
        }
    ],
}
html_context = {
    "github_user": "Novartis",
    "github_repo": "scar",
    "github_version": "develop",
    "doc_path": "docs",
}

# html_sidebars = {
#     "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
# }

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

sphinx_gallery_conf = {
    "examples_dirs": "tutorials",  # path to your example scripts
    "gallery_dirs": "_build/tutorial_gallery",  # path to where to save gallery generated output
    "filename_pattern": "/scAR_tutorial_",
#    "ignore_pattern": r"__init__\.py",
}