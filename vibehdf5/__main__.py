
import os
import sys
from pathlib import Path

from qtpy.QtWidgets import QApplication

from .hdf5_viewer import HDF5Viewer

def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    app = QApplication(argv)
    win = HDF5Viewer()

    # If a file path was passed as the first arg, open it
    if len(argv) > 1:
        candidate = argv[1]
        if os.path.isfile(candidate):
            win.load_hdf5(candidate)

    # Otherwise, if test file exists in workspace, open it by default
    else:
        default = Path(__file__).parent / "test_files.h5"
        if default.exists():
            win.load_hdf5(str(default))

    win.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
