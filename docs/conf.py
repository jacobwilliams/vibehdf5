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
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
]

templates_path = ['_templates']
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

# Common Qt method names to exclude
qt_method_names = {
    'blockSignals', 'childEvent', 'children', 'deleteLater', 'destroyed', 'disconnect',
    'dumpObjectInfo', 'dumpObjectTree', 'dynamicPropertyNames', 'event', 'eventFilter',
    'findChild', 'findChildren', 'inherits', 'installEventFilter', 'isSignalConnected',
    'isWidgetType', 'isWindowType', 'killTimer', 'metaObject', 'moveToThread', 'objectName',
    'objectNameChanged', 'parent', 'property', 'removeEventFilter', 'sender', 'senderSignalIndex',
    'setObjectName', 'setParent', 'setProperty', 'signalsBlocked', 'startTimer', 'thread',
    'timerEvent', 'tr', 'receivers', 'isSignalConnected',
    # QSyntaxHighlighter specific
    'setCurrentBlockState', 'setCurrentBlockUserData', 'setDocument', 'setFormat',
    'currentBlock', 'currentBlockState', 'currentBlockUserData', 'document',
    'previousBlockState', 'rehighlight', 'rehighlightBlock',
}

# Skip documenting members from Qt base classes
def skip_qt_members(app, what, name, obj, skip, options):
    """Skip members that come from Qt classes."""
    if skip:
        return True

    # Skip known Qt method names
    member_name = name.split('.')[-1]  # Get just the method name
    if member_name in qt_method_names:
        return True

    # Skip if the member's module is from Qt
    try:
        if hasattr(obj, '__module__'):
            module = obj.__module__ or ''
            if any(qt in module.lower() for qt in ['pyside6', 'pyqt5', 'pyqt6', 'qtpy', 'shiboken']):
                return True
    except Exception:
        pass

    # For methods and attributes, check if they're defined in the actual vibehdf5 source
    if what in ('method', 'attribute'):
        try:
            # Get the class that owns this member
            import inspect
            if hasattr(obj, '__objclass__'):
                owner_class = obj.__objclass__
            elif hasattr(obj, 'fget') and hasattr(obj.fget, '__objclass__'):
                owner_class = obj.fget.__objclass__
            else:
                # Try to infer from qualname
                if hasattr(obj, '__qualname__'):
                    parts = obj.__qualname__.split('.')
                    if len(parts) > 1:
                        # Check if this is from a vibehdf5 class
                        class_name = parts[-2]
                        # List of your actual classes
                        vibehdf5_classes = [
                            'HDF5Viewer', 'ColumnStatisticsDialog', 'ColumnSortDialog',
                            'ColumnFilterDialog', 'ColumnVisibilityDialog', 'UniqueValuesDialog',
                            'PlotOptionsDialog', 'DraggablePlotListWidget', 'ScaledImageLabel',
                            'DropTreeView', 'CustomSplitter'
                        ]
                        if class_name not in vibehdf5_classes:
                            return True
                        # Even if it's in the right class, check if defined in vibehdf5 module
                        if not obj.__module__.startswith('vibehdf5'):
                            return True
                return False

            # Check if the owner class is from Qt
            owner_module = owner_class.__module__
            if not owner_module.startswith('vibehdf5'):
                return True
        except Exception:
            pass

    return False

def setup(app):
    app.connect('autodoc-skip-member', skip_qt_members)

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# -- Options for autosummary -------------------------------------------------
autosummary_generate = True  # Automatically generate stub pages
autosummary_imported_members = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Autosummary settings
autosummary_generate = True
