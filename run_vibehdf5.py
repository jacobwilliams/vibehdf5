"""
Entry point for PyInstaller build.
This avoids issues with relative imports in __main__.py
"""
import sys
from vibehdf5.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
