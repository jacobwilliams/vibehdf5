"""VibeHDF5 main entry point."""

import os
import sys
from pathlib import Path

from qtpy.QtWidgets import QApplication

from .hdf5_viewer import HDF5Viewer


def main(argv: list[str] | None = None) -> int:
    """
    Launch the VibeHDF5 GUI application.

    Args:
        argv: Optional list of command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code from QApplication event loop (0 for normal exit).
    """

    argv = argv if argv is not None else sys.argv
    app = QApplication(argv)
    win = HDF5Viewer()

    # If a file path was passed as the first arg, open it
    if len(argv) > 1:
        candidate = argv[1]
        if os.path.isfile(candidate):
            win.load_hdf5(candidate)

    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
