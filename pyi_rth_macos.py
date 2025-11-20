"""
Runtime hook to fix Python library loading on macOS.
This ensures the Python shared library can be found at runtime.
"""

import sys
import os

# On macOS, ensure the library path is set correctly
if sys.platform == 'darwin':
    # Get the bundle directory
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS

        # Add the library path to DYLD_LIBRARY_PATH
        lib_path = os.path.join(bundle_dir, '_internal')
        if os.path.exists(lib_path):
            os.environ['DYLD_LIBRARY_PATH'] = lib_path + ':' + os.environ.get('DYLD_LIBRARY_PATH', '')
