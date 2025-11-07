from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QTreeView,
    QToolBar,
    QStatusBar,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QLabel,
    QPlainTextEdit,
)

from .hdf5_tree_model import HDF5TreeModel


# Helpers (placed before main/class so they are defined at runtime)
def _dataset_to_text(ds, limit_bytes: int = 1_000_000) -> tuple[str, str | None]:
    """Read an h5py dataset and return a text representation and an optional note.

    - If content exceeds limit_bytes, the output is truncated with a note.
    - Tries to decode bytes as UTF-8; falls back to hex preview for binary.
    """
    import numpy as np
    import h5py

    note = None
    # Best effort: read entire dataset (beware huge data)
    data = ds[()]

    # Convert to bytes if it's an array of fixed-length ASCII (S) blocks
    if isinstance(data, np.ndarray) and data.dtype.kind == 'S':
        try:
            # Flatten and join bytes chunks
            b = b''.join(x.tobytes() if hasattr(x, 'tobytes') else bytes(x) for x in data.ravel())
        except Exception:
            b = data.tobytes()
        return _bytes_to_text(b, limit_bytes)

    # Variable length strings
    vld = h5py.check_string_dtype(ds.dtype)
    if vld is not None:
        try:
            # Read as Python str
            as_str = ds.asstr()[()]
            if isinstance(as_str, np.ndarray):
                text = '\n'.join(map(str, as_str.ravel().tolist()))
            else:
                text = str(as_str)
            note = None
            if len(text.encode('utf-8')) > limit_bytes:
                enc = text.encode('utf-8')[:limit_bytes]
                text = enc.decode('utf-8', errors='ignore') + "\n… (truncated)"
                note = f"Preview limited to {limit_bytes} bytes"
            return text, note
        except Exception:
            pass

    # Raw bytes
    if isinstance(data, (bytes, bytearray, np.void)):
        return _bytes_to_text(bytes(data), limit_bytes)

    # Numeric or other arrays: show a compact preview
    if isinstance(data, np.ndarray):
        flat = data.ravel()
        preview_count = min(2000, flat.size)
        text = np.array2string(flat[:preview_count], threshold=preview_count)
        note = None
        if flat.size > preview_count:
            note = f"Showing first {preview_count} elements out of {flat.size}"
        return text, note

    # Fallback to repr
    t = repr(data)
    if len(t) > 200_000:
        t = t[:200_000] + '… (truncated)'
        note = "Preview truncated"
    return t, note


def _bytes_to_text(b: bytes, limit_bytes: int = 1_000_000) -> tuple[str, str | None]:
    note = None
    if len(b) > limit_bytes:
        b = b[:limit_bytes]
        note = f"Preview limited to {limit_bytes} bytes"
    try:
        return b.decode('utf-8'), note
    except UnicodeDecodeError:
        # Provide a hex dump preview
        import binascii
        hexstr = binascii.hexlify(b).decode('ascii')
        # Group hex bytes in pairs for readability
        grouped = ' '.join(hexstr[i:i+2] for i in range(0, len(hexstr), 2))
        if len(grouped) > 200_000:
            grouped = grouped[:200_000] + '… (truncated)'
            note = "Preview truncated"
        return grouped, (note or "binary data shown as hex")


class HDF5Viewer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF5 Viewer")
        self.resize(900, 600)

        # Central widget: splitter with tree (left) and preview (right)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        splitter = QSplitter(self)
        central_layout.addWidget(splitter)

        # Tree view + model (left)
        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeView(left)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionBehavior(QTreeView.SelectRows)
        self.tree.setHeaderHidden(False)
        left_layout.addWidget(self.tree)
        splitter.addWidget(left)

        self.model = HDF5TreeModel(self)
        self.tree.setModel(self.model)
        self.tree.header().setStretchLastSection(True)
        self.tree.header().setDefaultSectionSize(350)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)

        # Preview panel (right)
        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self.preview_label = QLabel("No selection")
        self.preview_edit = QPlainTextEdit(self)
        self.preview_edit.setReadOnly(True)
        # Use a fixed-width font for better alignment of file contents
        try:
            fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        except Exception:
            fixed = QFont("Courier New")
        self.preview_edit.setFont(fixed)
        # Avoid wrapping so columns/bytes stay aligned
        try:
            self.preview_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        except Exception:
            pass
        right_layout.addWidget(self.preview_label)
        right_layout.addWidget(self.preview_edit)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Tool bar / actions
        self._create_actions()
        self._create_toolbar()
        self.setStatusBar(QStatusBar(self))
        self.tree.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def _create_actions(self) -> None:
        self.act_open = QAction("Open HDF5…", self)
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.triggered.connect(self.open_file_dialog)

        self.act_expand = QAction("Expand All", self)
        self.act_expand.triggered.connect(self.tree.expandAll)

        self.act_collapse = QAction("Collapse All", self)
        self.act_collapse.triggered.connect(self.tree.collapseAll)

        self.act_quit = QAction("Quit", self)
        self.act_quit.setShortcut("Ctrl+Q")
        self.act_quit.triggered.connect(self.close)

    def _create_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        self.addToolBar(tb)
        tb.addAction(self.act_open)
        tb.addSeparator()
        tb.addAction(self.act_expand)
        tb.addAction(self.act_collapse)
        tb.addSeparator()
        tb.addAction(self.act_quit)

    def open_file_dialog(self) -> None:
        last_dir = os.getcwd()
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 File",
            last_dir,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if filepath:
            self.load_hdf5(filepath)

    def load_hdf5(self, path: str | Path) -> None:
        path = str(path)
        try:
            self.model.load_file(path)
        except Exception as exc:  # show friendly error dialog
            QMessageBox.critical(
                self,
                "Failed to open HDF5",
                f"Could not open file:\n{path}\n\n{exc}",
            )
            return
        self.statusBar().showMessage(path)
        self.tree.expandToDepth(1)
        self.preview_label.setText("No selection")
        self.preview_edit.setPlainText("")

    # Selection handling
    def on_selection_changed(self, selected, _deselected) -> None:
        indexes = selected.indexes()
        if not indexes:
            return
        index = indexes[0]
        item = self.model.itemFromIndex(index)
        kind = item.data(self.model.ROLE_KIND)
        path = item.data(self.model.ROLE_PATH)

        if kind == "dataset":
            self.preview_dataset(path)
        elif kind == "attr":
            key = item.data(self.model.ROLE_ATTR_KEY)
            self.preview_attribute(path, key)
        elif kind == "group":
            self.preview_label.setText(f"Group: {path}")
            self.preview_edit.setPlainText("(No content to display)")
        else:
            self.preview_label.setText(str(kind) if kind else "")
            self.preview_edit.setPlainText("")

    def preview_dataset(self, dspath: str) -> None:
        self.preview_label.setText(f"Dataset: {os.path.basename(dspath)}")
        fpath = self.model.filepath
        if not fpath:
            self.preview_edit.setPlainText("No file loaded")
            return
        try:
            import h5py
            with h5py.File(fpath, "r") as h5:
                obj = h5[dspath]
                if not isinstance(obj, h5py.Dataset):
                    self.preview_edit.setPlainText("Selected path is not a dataset.")
                    return
                ds = obj
                text, note = _dataset_to_text(ds, limit_bytes=1_000_000)
                header = f"shape={ds.shape}, dtype={ds.dtype}"
                if note:
                    header += f"\n{note}"
                self.preview_edit.setPlainText(header + "\n\n" + text)
        except Exception as exc:
            self.preview_edit.setPlainText(f"Error reading dataset:\n{exc}")

    def preview_attribute(self, grouppath: str, key: str) -> None:
        self.preview_label.setText(f"Attribute: {grouppath}@{key}")
        fpath = self.model.filepath
        if not fpath:
            self.preview_edit.setPlainText("No file loaded")
            return
        try:
            import h5py
            with h5py.File(fpath, "r") as h5:
                g = h5[grouppath]
                val = g.attrs[key]
                self.preview_edit.setPlainText(repr(val))
        except Exception as exc:
            self.preview_edit.setPlainText(f"Error reading attribute:\n{exc}")


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
