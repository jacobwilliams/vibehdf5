"""
Entry point for PyInstaller build.
This avoids issues with relative imports in __main__.py
"""
import sys
from vibehdf5.hdf5_viewer import __main__

if __name__ == "__main__":
    sys.exit(__main__.main())
