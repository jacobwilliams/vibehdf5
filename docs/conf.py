# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'vibehdf5'
copyright = '2025, Jacob Williams'
author = 'Jacob Williams'

try:
    from vibehdf5 import __version__
    version = __version__
    release = __version__
except ImportError:
    version = '1.1.0'
    release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output -----------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# -- Autodoc configuration -----------------------------------------------------

# Mock Qt to prevent import errors (your code imports qtpy)
autodoc_mock_imports = ['PySide6', 'PyQt5', 'PyQt6', 'qtpy']

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# -- Napoleon configuration (for Google/NumPy style docstrings) -------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Intersphinx configuration (for linking to other docs) ------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}
