from __future__ import annotations

import os
import sys
from pathlib import Path

from qtpy.QtCore import Qt, QUrl, QMimeData
from qtpy.QtGui import QAction, QFont, QFontDatabase, QDrag
from qtpy.QtWidgets import (
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
    QMenu,
)

from .hdf5_tree_model import HDF5TreeModel
from .utilities import excluded_dirs, excluded_files


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


class ScaledImageLabel(QLabel):
    def __init__(self, parent=None, rescale_callback=None):
        super().__init__(parent)
        self._rescale_callback = rescale_callback

    def resizeEvent(self, event):
        if self._rescale_callback:
            self._rescale_callback()
        super().resizeEvent(event)

class HDF5Viewer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF5 Viewer")
        self.resize(900, 600)
        self._original_pixmap = None

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

        # Enable drag-and-drop
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QTreeView.DragOnly)

        # Context menu on tree
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.on_tree_context_menu)

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

        # Image preview label (hidden by default)
        self.preview_image = ScaledImageLabel(self, rescale_callback=self._show_scaled_image)
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setVisible(False)
        self.preview_image.setScaledContents(False)  # We'll scale manually for aspect ratio
        right_layout.addWidget(self.preview_image)
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

        # Add files/folder actions
        self.act_add_files = QAction("Add Files…", self)
        self.act_add_files.setShortcut("Ctrl+Shift+F")
        self.act_add_files.triggered.connect(self.add_files_dialog)

        self.act_add_folder = QAction("Add Folder…", self)
        self.act_add_folder.setShortcut("Ctrl+Shift+D")
        self.act_add_folder.triggered.connect(self.add_folder_dialog)

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
        tb.addAction(self.act_add_files)
        tb.addAction(self.act_add_folder)
        tb.addSeparator()
        tb.addAction(self.act_expand)
        tb.addAction(self.act_collapse)
        tb.addSeparator()
        tb.addAction(self.act_quit)

    # Determine where to add new content in the HDF5 file
    def _get_target_group_path(self) -> str:
        sel = self.tree.selectionModel().selectedIndexes()
        if not sel:
            return "/"
        index = sel[0].sibling(sel[0].row(), 0)
        item = self.model.itemFromIndex(index)
        if item is None:
            return "/"
        kind = item.data(self.model.ROLE_KIND)
        path = item.data(self.model.ROLE_PATH) or "/"
        if kind == "group":
            return path
        if kind in ("attr", "attrs-folder"):
            return path  # path points to the owner group/dataset for attrs-folder and attr
        if kind == "dataset":
            # parent group of dataset
            try:
                import posixpath
                return posixpath.dirname(path) or "/"
            except Exception:
                return "/"
        if kind == "file":
            return "/"
        return "/"

    def add_files_dialog(self) -> None:
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.information(self, "No file", "Open an HDF5 file first.")
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Select files to add")
        if not files:
            return
        target_group = self._get_target_group_path()
        errors: list[str] = []
        added = 0
        try:
            import h5py
            import numpy as np
            import posixpath
            with h5py.File(fpath, "r+") as h5:
                grp = h5
                if target_group != "/":
                    grp = h5.require_group(target_group)
                for path_on_disk in files:
                    name = os.path.basename(path_on_disk)
                    if name in excluded_files:
                        continue
                    h5_path = posixpath.join(target_group, name) if target_group != "/" else "/" + name
                    try:
                        self._create_dataset_from_file(grp, h5_path, path_on_disk, np)
                        added += 1
                    except FileExistsError:
                        # Ask to overwrite
                        resp = QMessageBox.question(
                            self,
                            "Overwrite?",
                            f"'{h5_path}' exists. Overwrite?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No,
                        )
                        if resp == QMessageBox.Yes:
                            del h5[h5_path]
                            self._create_dataset_from_file(grp, h5_path, path_on_disk, np)
                            added += 1
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Add files failed", str(exc))
            return
        # Refresh
        self.model.load_file(fpath)
        self.tree.expandToDepth(1)
        if errors:
            QMessageBox.warning(self, "Completed with errors", "\n".join(errors))
        elif added:
            self.statusBar().showMessage(f"Added {added} file(s) to {target_group}", 5000)

    def add_folder_dialog(self) -> None:
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.information(self, "No file", "Open an HDF5 file first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select folder to add")
        if not directory:
            return
        target_group = self._get_target_group_path()
        errors: list[str] = []
        added = 0
        try:
            import h5py
            import numpy as np
            import posixpath
            with h5py.File(fpath, "r+") as h5:
                base_name = os.path.basename(os.path.normpath(directory))
                root_h5_group = target_group
                if root_h5_group == "/":
                    root_h5_group = "/" + base_name
                else:
                    root_h5_group = posixpath.join(root_h5_group, base_name)
                # walk
                for dirpath, dirnames, filenames in os.walk(directory):
                    # prune excluded dirs
                    dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
                    rel = os.path.relpath(dirpath, directory)
                    rel = "." if rel == "." else rel.replace("\\", "/")
                    current_group_path = root_h5_group if rel == "." else posixpath.join(root_h5_group, rel)
                    grp = h5.require_group(current_group_path)
                    for filename in filenames:
                        if filename in excluded_files:
                            continue
                        file_on_disk = os.path.join(dirpath, filename)
                        h5_path = posixpath.join(current_group_path, filename)
                        try:
                            self._create_dataset_from_file(grp, h5_path, file_on_disk, np)
                            added += 1
                        except FileExistsError:
                            resp = QMessageBox.question(
                                self,
                                "Overwrite?",
                                f"'{h5_path}' exists. Overwrite?",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No,
                            )
                            if resp == QMessageBox.Yes:
                                del h5[h5_path]
                                self._create_dataset_from_file(grp, h5_path, file_on_disk, np)
                                added += 1
                        except Exception as exc:  # noqa: BLE001
                            errors.append(f"{h5_path}: {exc}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Add folder failed", str(exc))
            return
        # Refresh
        self.model.load_file(fpath)
        self.tree.expandToDepth(2)
        if errors:
            QMessageBox.warning(self, "Completed with errors", "\n".join(errors))
        elif added:
            self.statusBar().showMessage(f"Added {added} item(s) under {target_group}", 5000)

    def _create_dataset_from_file(self, grp, h5_path: str, disk_path: str, np) -> None:
        """Create a dataset at h5_path from a file on disk under the given group (or file root).

        Raises FileExistsError if the path already exists.
        """
        import h5py
        # Check existence
        f = grp.file
        if h5_path in f:
            raise FileExistsError(h5_path)
        # Ensure parent groups exist
        parent = os.path.dirname(h5_path).replace("\\", "/")
        if parent and parent != "/":
            f.require_group(parent)
        # Try text then binary
        try:
            with open(disk_path, "r", encoding="utf-8") as fin:
                data = fin.read()
            f.create_dataset(h5_path, data=data, dtype=h5py.string_dtype(encoding="utf-8"))
            return
        except Exception:  # noqa: BLE001
            pass
        with open(disk_path, "rb") as fin:
            bdata = fin.read()
        f.create_dataset(h5_path, data=np.frombuffer(bdata, dtype="uint8"))

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

    # Context menu handling
    def on_tree_context_menu(self, point) -> None:
        index = self.tree.indexAt(point)
        if not index.isValid():
            return
        # Always act on column 0 item for role data
        index0 = index.siblingAtColumn(0)
        item = self.model.itemFromIndex(index0)
        if item is None:
            return
        kind = item.data(self.model.ROLE_KIND)
        path = item.data(self.model.ROLE_PATH)
        attr_key = item.data(self.model.ROLE_ATTR_KEY)

        # Determine if deletable and label
        deletable = False
        label = None
        if kind == "dataset":
            deletable = True
            label = f"Delete dataset '{item.text()}'"
        elif kind == "group":
            # Don't allow deleting the file root
            if path and path != "/":
                deletable = True
                label = f"Delete group '{item.text()}'"
        elif kind == "attr":
            deletable = True
            label = f"Delete attribute '{attr_key}'"

        menu = QMenu(self)
        if deletable and label:
            act_delete = menu.addAction(label)
        else:
            # Nothing to act on
            return

        global_pos = self.tree.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)
        if chosen == act_delete:
            # Confirm destructive action
            target_desc = label.replace("Delete ", "") if label else "item"
            resp = QMessageBox.question(
                self,
                "Confirm delete",
                f"Are you sure you want to delete {target_desc}?\n\nThis will modify the HDF5 file and cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                self._perform_delete(kind, path, attr_key)

    def _perform_delete(self, kind: str, path: str, attr_key: str | None) -> None:
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.warning(self, "No file", "No HDF5 file is loaded.")
            return
        try:
            import h5py
            with h5py.File(fpath, "r+") as h5:
                if kind == "dataset":
                    # Deleting a dataset link by absolute path
                    del h5[path]
                elif kind == "group":
                    if path == "/":
                        raise ValueError("Cannot delete the root group")
                    del h5[path]
                elif kind == "attr":
                    if attr_key is None:
                        raise ValueError("Missing attribute key")
                    # For attributes, 'path' is the group/dataset owner path
                    owner = h5[path]
                    del owner.attrs[attr_key]
                else:
                    raise ValueError(f"Unsupported kind: {kind}")
        except Exception as exc:
            QMessageBox.critical(self, "Delete failed", f"Could not delete: {exc}")
            return

        # Refresh model to reflect changes
        try:
            self.model.load_file(fpath)
            self.tree.expandToDepth(1)
        except Exception as exc:
            QMessageBox.warning(self, "Refresh failed", f"Deleted, but failed to refresh view: {exc}")

    def preview_dataset(self, dspath: str) -> None:
        self.preview_label.setText(f"Dataset: {os.path.basename(dspath)}")
        fpath = self.model.filepath
        if not fpath:
            self.preview_edit.setPlainText("No file loaded")
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
            return
        # If the dataset name ends with .png, try to display as image
        if dspath.lower().endswith('.png'):
            try:
                import h5py
                from qtpy.QtGui import QPixmap
                with h5py.File(fpath, "r") as h5:
                    obj = h5[dspath]
                    if not isinstance(obj, h5py.Dataset):
                        self.preview_edit.setPlainText("Selected path is not a dataset.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        return
                    # Read raw bytes from dataset
                    data = obj[()]
                    if isinstance(data, bytes):
                        img_bytes = data
                    elif hasattr(data, 'tobytes'):
                        img_bytes = data.tobytes()
                    else:
                        self.preview_edit.setPlainText("Dataset is not a valid PNG byte array.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        return
                    pixmap = QPixmap()
                    if pixmap.loadFromData(img_bytes, "PNG"):
                        # Scale pixmap to fit preview area, maintaining aspect ratio
                        self._show_scaled_image(pixmap)
                        self.preview_image.setVisible(True)
                        self.preview_edit.setVisible(False)
                    else:
                        self.preview_edit.setPlainText("Failed to load PNG image from dataset.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
            except Exception as exc:
                self.preview_edit.setPlainText(f"Error reading PNG dataset:\n{exc}")
                self.preview_edit.setVisible(True)
                self.preview_image.setVisible(False)
            return
        # Otherwise, show text preview for non-PNG datasets
        try:
            import h5py
            with h5py.File(fpath, "r") as h5:
                obj = h5[dspath]
                if not isinstance(obj, h5py.Dataset):
                    self.preview_edit.setPlainText("Selected path is not a dataset.")
                    self.preview_edit.setVisible(True)
                    self.preview_image.setVisible(False)
                    return
                ds = obj
                text, note = _dataset_to_text(ds, limit_bytes=1_000_000)
                header = f"shape={ds.shape}, dtype={ds.dtype}"
                if note:
                    header += f"\n{note}"
                self.preview_edit.setPlainText(header + "\n\n" + text)
                self.preview_edit.setVisible(True)
                self.preview_image.setVisible(False)
        except Exception as exc:
            self.preview_edit.setPlainText(f"Error reading dataset:\n{exc}")
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
    def _show_scaled_image(self, pixmap=None):
        # Use the provided pixmap or the stored one
        if pixmap is not None:
            self._original_pixmap = pixmap
        pixmap = self._original_pixmap
        if pixmap is None:
            return
        label_size = self.preview_image.size()
        if label_size.width() < 10 or label_size.height() < 10:
            label_size = self.preview_image.parentWidget().size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_image.setPixmap(scaled)

    def resizeEvent(self, event):
        # If an image is visible, rescale it to fit the new size
        if self.preview_image.isVisible():
            self._show_scaled_image()
        super().resizeEvent(event)

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
