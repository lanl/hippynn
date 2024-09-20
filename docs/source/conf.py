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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "hippynn"
copyright = "2019, Los Alamos National Laboratory"
author = "Nicholas Lubbers et al"

# The full version, including alpha/beta/rc tags
import os
os.environ['HIPPYNN_USE_CUSTOM_KERNELS'] = "False"
import hippynn

release = hippynn.__version__
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx_rtd_theme", "sphinx.ext.viewcode", 'sphinx.ext.autosummary', 'sphinxcontrib.bibtex']
bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# Autodoc options

# The following are optional, so we mock them for doc purposes.
# TODO: Can we programmatically get these from our list of optional dependencies?
autodoc_mock_imports = [
    "ase",
    "matplotlib",
    "h5py",
    "seqm",
    "schnetpack",
    "triton",
    "numba",
    "cupy",
    "lammps",
    "pytorch_lightning",
    "scipy",
    "graphviz",
]

autodoc_default_options = {
    "no-show-inheritance": True,
    # ignore-module-all is FALSE by default, which we currently prefer.
    # note to the future:
    #   don't -set- ignore-module-all to False, as it will still include the directive argument.
}
autodoc_member_order = 'groupwise'
add_module_names = False

# Autosummary Options
autosummary_mock_imports = autodoc_mock_imports
autosummary_imported_members = False
autosummary_ignore_module_all = True  # This is intentionally different from autodoc config.


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Just test to error if this doesn't exist.
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": -1,
    "prev_next_buttons_location": "both",
    "navigation_with_keys": True,
    "sticky_navigation": False,
    "style_external_links": True,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

# html_logo = "_static/logo.svg"
