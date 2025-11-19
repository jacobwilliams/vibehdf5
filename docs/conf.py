# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'vibehdf5'
copyright = '2025, Jacob Williams'
author = 'Jacob Williams'

# The version info for the project you're documenting
try:
    from vibehdf5 import __version__
    version = __version__
    release = __version__
except ImportError:
    version = '0.1.0'
    release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'inherited-members': False,  # Don't show inherited members
}

# Don't document inherited members from base classes
autodoc_inherit_docstrings = False

# Only document members defined in the module itself
autodoc_class_signature = 'separated'

# Mock Qt imports to prevent documenting Qt base classes
autodoc_mock_imports = ['PySide6', 'PyQt5', 'PyQt6', 'qtpy']

# Configure mocked objects to support Qt operations
class MockQt:
    """Mock Qt namespace that supports arithmetic operations."""
    UserRole = 0x0100  # Qt.UserRole value
    
autodoc_mock_imports = ['PySide6', 'PyQt5', 'PyQt6']

# Setup mock for qtpy that handles Qt.UserRole properly
import sys
from unittest.mock import MagicMock

class QtMock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        if name == 'UserRole':
            return 0x0100  # Return an int that can be used in arithmetic
        return MagicMock()

sys.modules['qtpy'] = MagicMock()
sys.modules['qtpy.QtCore'] = MagicMock()
sys.modules['qtpy.QtGui'] = MagicMock()
sys.modules['qtpy.QtWidgets'] = MagicMock()
sys.modules['qtpy.QtCore'].Qt = QtMock()

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

