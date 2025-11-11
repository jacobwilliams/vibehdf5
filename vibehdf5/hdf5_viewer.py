from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import posixpath
import binascii
import gzip

from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QFont, QFontDatabase, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QTreeView,
    QAbstractItemView,
    QToolBar,
    QStatusBar,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QPlainTextEdit,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QProgressDialog,
    QDialog,
    QDialogButtonBox,
    QComboBox,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QFrame,
    QTabWidget,
)

from .hdf5_tree_model import HDF5TreeModel
from .utilities import excluded_dirs, excluded_files
from .syntax_highlighter import SyntaxHighlighter, get_language_from_path


# Helpers (placed before main/class so they are defined at runtime)
def _sanitize_hdf5_name(name: str) -> str:
    """Sanitize a name for use as an HDF5 dataset/group member.

    - replaces '/' with '_' (since '/' is the HDF5 path separator)
    - strips leading/trailing whitespace
    - returns 'unnamed' if the result is empty
    """
    try:
        s = (name or "").strip()
        s = s.replace("/", "_")
        return s or "unnamed"
    except Exception:  # noqa: BLE001
        return "unnamed"

def _dataset_to_text(ds, limit_bytes: int = 1_000_000) -> tuple[str, str | None]:
    """Read an h5py dataset and return a text representation and an optional note.

    - If content exceeds limit_bytes, the output is truncated with a note.
    - Tries to decode bytes as UTF-8; falls back to hex preview for binary.
    - Automatically decompresses gzip-compressed text datasets.
    """

    note = None
    # Best effort: read entire dataset (beware huge data)
    data = ds[()]

    # Check if this is a gzip-compressed text dataset
    try:
        if 'compressed' in ds.attrs and ds.attrs['compressed'] == 'gzip':
            if isinstance(data, np.ndarray) and data.dtype == np.uint8:
                compressed_bytes = data.tobytes()
                decompressed = gzip.decompress(compressed_bytes)
                encoding = ds.attrs.get('original_encoding', 'utf-8')
                if isinstance(encoding, bytes):
                    encoding = encoding.decode('utf-8')
                # Check if this is binary data
                if encoding == 'binary':
                    # Return decompressed binary data for further processing
                    return _bytes_to_text(decompressed, limit_bytes, decompressed=True)
                # Otherwise it's text
                text = decompressed.decode(encoding)
                if len(text) > limit_bytes:
                    text = text[:limit_bytes] + "\n… (truncated)"
                    note = f"Preview limited to {limit_bytes} characters (decompressed)"
                else:
                    note = "(decompressed from gzip)"
                return text, note
    except Exception:  # noqa: BLE001
        pass

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


def _bytes_to_text(b: bytes, limit_bytes: int = 1_000_000, decompressed: bool = False) -> tuple[str, str | None]:
    note = None
    if len(b) > limit_bytes:
        b = b[:limit_bytes]
        note = f"Preview limited to {limit_bytes} bytes"
        if decompressed:
            note = f"Preview limited to {limit_bytes} bytes (decompressed)"
    elif decompressed:
        note = "(decompressed from gzip)"
    try:
        return b.decode('utf-8'), note
    except UnicodeDecodeError:
        # Provide a hex dump preview
        hexstr = binascii.hexlify(b).decode('ascii')
        # Group hex bytes in pairs for readability
        grouped = ' '.join(hexstr[i:i+2] for i in range(0, len(hexstr), 2))
        if len(grouped) > 200_000:
            grouped = grouped[:200_000] + '… (truncated)'
            note = "Preview truncated"
        suffix = " (decompressed)" if decompressed else ""
        return grouped, ((note or "binary data shown as hex") + suffix)


class ScaledImageLabel(QLabel):
    def __init__(self, parent=None, rescale_callback=None):
        super().__init__(parent)
        self._rescale_callback = rescale_callback

    def resizeEvent(self, event):
        if self._rescale_callback:
            self._rescale_callback()
        super().resizeEvent(event)


class DropTreeView(QTreeView):
    """TreeView that accepts external file/folder drops and forwards to the viewer."""

    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md and md.hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        md = event.mimeData()
        if md and md.hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        md = event.mimeData()
        if not (self.viewer and md):
            return super().dropEvent(event)

        # Check if this is an internal HDF5 move
        if md.hasFormat("application/x-hdf5-path"):
            source_path = bytes(md.data("application/x-hdf5-path")).decode('utf-8')
            try:
                pos = event.position().toPoint()
            except Exception:  # noqa: BLE001
                pos = event.pos()
            idx = self.indexAt(pos)
            target_group = self.viewer._get_target_group_path_for_index(idx)

            # Perform the move operation
            success = self.viewer._move_hdf5_item(source_path, target_group)
            if success:
                fpath = self.viewer.model.filepath
                if fpath:
                    self.viewer.model.load_file(fpath)
                    self.viewer.tree.expandToDepth(2)
                self.viewer.statusBar().showMessage(
                    f"Moved '{source_path}' to '{target_group}'", 5000
                )
                event.acceptProposedAction()
            else:
                event.ignore()
            return

        # Handle external file drops
        if md.hasUrls():
            urls = md.urls()
            paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
            files = [p for p in paths if os.path.isfile(p)]
            folders = [p for p in paths if os.path.isdir(p)]
            try:
                pos = event.position().toPoint()
            except Exception:  # noqa: BLE001
                pos = event.pos()
            idx = self.indexAt(pos)
            target_group = self.viewer._get_target_group_path_for_index(idx)
            added, errors = self.viewer._add_items_batch(files, folders, target_group)
            fpath = self.viewer.model.filepath
            if fpath:
                self.viewer.model.load_file(fpath)
                self.viewer.tree.expandToDepth(2)
            if errors:
                QMessageBox.warning(self, "Completed with errors", "\n".join(errors))
            elif added:
                self.viewer.statusBar().showMessage(
                    f"Added {added} item(s) under {target_group}", 5000
                )
            event.acceptProposedAction()
            return

        super().dropEvent(event)


class ColumnFilterDialog(QDialog):
    """Dialog for configuring column filters."""

    def __init__(self, column_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Column Filters")
        self.resize(600, 400)

        self.column_names = column_names
        self.filters = []  # List of (column_name, operator, value) tuples

        layout = QVBoxLayout(self)

        # Instructions
        info_label = QLabel("Add filters to show only rows matching the criteria:")
        layout.addWidget(info_label)

        # Scroll area for filters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.StyledPanel)

        filter_container = QWidget()
        self.filter_layout = QVBoxLayout(filter_container)
        self.filter_layout.setContentsMargins(5, 5, 5, 5)
        self.filter_layout.addStretch()

        scroll.setWidget(filter_container)
        layout.addWidget(scroll)

        # Add filter button
        add_btn = QPushButton("+ Add Filter")
        add_btn.clicked.connect(self._add_filter_row)
        layout.addWidget(add_btn)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _add_filter_row(self, col_name=None, operator="==", value=""):
        """Add a new filter row to the dialog."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        # Column dropdown
        col_combo = QComboBox()
        col_combo.addItems(self.column_names)
        if col_name and col_name in self.column_names:
            col_combo.setCurrentText(col_name)
        col_combo.setMinimumWidth(150)

        # Operator dropdown
        op_combo = QComboBox()
        op_combo.addItems(["==", "!=", ">", ">=", "<", "<=", "contains", "startswith", "endswith"])
        op_combo.setCurrentText(operator)
        op_combo.setMinimumWidth(100)

        # Value input
        value_edit = QLineEdit(value)
        value_edit.setPlaceholderText("Filter value...")
        value_edit.setMinimumWidth(150)

        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_filter_row(row_widget))

        row_layout.addWidget(QLabel("Column:"))
        row_layout.addWidget(col_combo)
        row_layout.addWidget(QLabel("Operator:"))
        row_layout.addWidget(op_combo)
        row_layout.addWidget(QLabel("Value:"))
        row_layout.addWidget(value_edit)
        row_layout.addWidget(remove_btn)
        row_layout.addStretch()

        # Store references for later retrieval
        row_widget._col_combo = col_combo
        row_widget._op_combo = op_combo
        row_widget._value_edit = value_edit

        # Insert before the stretch
        self.filter_layout.insertWidget(self.filter_layout.count() - 1, row_widget)

    def _remove_filter_row(self, row_widget):
        """Remove a filter row from the dialog."""
        self.filter_layout.removeWidget(row_widget)
        row_widget.deleteLater()

    def get_filters(self):
        """Return list of active filters as (column_name, operator, value) tuples."""
        filters = []
        for i in range(self.filter_layout.count() - 1):  # -1 to skip stretch
            widget = self.filter_layout.itemAt(i).widget()
            if widget and hasattr(widget, '_col_combo'):
                col_name = widget._col_combo.currentText()
                operator = widget._op_combo.currentText()
                value = widget._value_edit.text()
                if value:  # Only include filters with values
                    filters.append((col_name, operator, value))
        return filters

    def set_filters(self, filters):
        """Set initial filters from a list of (column_name, operator, value) tuples."""
        for col_name, operator, value in filters:
            self._add_filter_row(col_name, operator, value)


class HDF5Viewer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF5 Viewer")
        self.resize(900, 600)
        self._original_pixmap = None
        self._current_highlighter = None  # Track current syntax highlighter

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
        self.tree = DropTreeView(left, viewer=self)
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

        # Enable drag-and-drop (both export and external import)
        self.tree.setDragEnabled(True)  # allow dragging out
        self.tree.setAcceptDrops(True)  # allow external drops
        self.tree.setDragDropMode(QAbstractItemView.DragDrop)
        self.tree.setDefaultDropAction(Qt.MoveAction)  # move for internal, copy for external

        # Context menu on tree
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.on_tree_context_menu)

        # Preview panel (right)
        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Create a vertical splitter for content and attributes
        right_splitter = QSplitter(Qt.Vertical, right)
        right_layout.addWidget(right_splitter)

        # Top section: main content preview
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

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
        content_layout.addWidget(self.preview_label)
        content_layout.addWidget(self.preview_edit)

        # Filter panel for CSV tables (hidden by default)
        self.filter_panel = QWidget()
        filter_panel_layout = QHBoxLayout(self.filter_panel)
        filter_panel_layout.setContentsMargins(5, 5, 5, 5)

        filter_label = QLabel("Filters:")
        filter_panel_layout.addWidget(filter_label)

        self.filter_status_label = QLabel("No filters applied")
        filter_panel_layout.addWidget(self.filter_status_label)

        self.btn_configure_filters = QPushButton("Configure Filters...")
        self.btn_configure_filters.clicked.connect(self._configure_filters_dialog)
        filter_panel_layout.addWidget(self.btn_configure_filters)

        self.btn_clear_filters = QPushButton("Clear Filters")
        self.btn_clear_filters.clicked.connect(self._clear_filters)
        self.btn_clear_filters.setEnabled(False)
        filter_panel_layout.addWidget(self.btn_clear_filters)

        filter_panel_layout.addStretch()

        self.filter_panel.setVisible(False)
        content_layout.addWidget(self.filter_panel)

        # Table widget for CSV/tabular data (hidden by default)
        self.preview_table = QTableWidget(self)
        self.preview_table.setVisible(False)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        # Enable selecting multiple columns for plotting
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.preview_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.preview_table.itemSelectionChanged.connect(self._update_plot_action_enabled)
        content_layout.addWidget(self.preview_table)

        # Image preview label (hidden by default)
        self.preview_image = ScaledImageLabel(self, rescale_callback=self._show_scaled_image)
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setVisible(False)
        self.preview_image.setScaledContents(False)  # We'll scale manually for aspect ratio
        content_layout.addWidget(self.preview_image)

        right_splitter.addWidget(content_widget)

        # Bottom section: tabbed widget for Attributes and Plot
        self.bottom_tabs = QTabWidget()

        # Attributes tab
        attrs_widget = QWidget()
        attrs_layout = QVBoxLayout(attrs_widget)
        attrs_layout.setContentsMargins(0, 0, 0, 0)

        self.attrs_label = QLabel("Attributes")
        self.attrs_label.setVisible(False)
        self.attrs_table = QTableWidget(self)
        self.attrs_table.setVisible(True)  # Always visible when tab is active
        self.attrs_table.setColumnCount(2)
        self.attrs_table.setHorizontalHeaderLabels(["Name", "Value"])
        self.attrs_table.horizontalHeader().setStretchLastSection(True)
        self.attrs_table.setAlternatingRowColors(True)
        self.attrs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.attrs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        attrs_layout.addWidget(self.attrs_table)

        self.bottom_tabs.addTab(attrs_widget, "Attributes")

        # Plot tab
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Import matplotlib components for embedding
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            self.plot_figure = Figure(figsize=(8, 6))
            self.plot_canvas = FigureCanvas(self.plot_figure)
            plot_layout.addWidget(self.plot_canvas)
            self._matplotlib_available = True
        except ImportError:
            # Matplotlib not available
            no_plot_label = QLabel("Matplotlib not installed. Install it to enable plotting.")
            no_plot_label.setAlignment(Qt.AlignCenter)
            plot_layout.addWidget(no_plot_label)
            self.plot_figure = None
            self.plot_canvas = None
            self._matplotlib_available = False

        self.bottom_tabs.addTab(plot_widget, "Plot")

        # Start with Attributes tab visible
        self.bottom_tabs.setCurrentIndex(0)

        right_splitter.addWidget(self.bottom_tabs)

        # Set initial sizes: main content gets most of the space
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Tool bar / actions
        self._create_actions()
        self._create_toolbar()
        self.setStatusBar(QStatusBar(self))
        self.tree.selectionModel().selectionChanged.connect(self.on_selection_changed)

        # Track currently previewed CSV group (for plotting)
        self._current_csv_group_path: str | None = None

        # Track CSV data and filters
        self._csv_data_dict: dict[str, np.ndarray] = {}  # Full unfiltered data
        self._csv_column_names: list[str] = []
        self._csv_filters: list[tuple[str, str, str]] = []  # (column, operator, value)
        self._csv_filtered_indices: np.ndarray | None = None  # Indices of visible rows

    def _create_actions(self) -> None:
        self.act_new = QAction("New HDF5 File…", self)
        self.act_new.setShortcut("Ctrl+N")
        self.act_new.triggered.connect(self.new_file_dialog)

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

        self.act_new_folder = QAction("New Folder…", self)
        self.act_new_folder.setShortcut("Ctrl+Shift+N")
        self.act_new_folder.triggered.connect(self.new_folder_dialog)

        self.act_expand = QAction("Expand All", self)
        self.act_expand.triggered.connect(self.tree.expandAll)

        self.act_collapse = QAction("Collapse All", self)
        self.act_collapse.triggered.connect(self.tree.collapseAll)

        self.act_quit = QAction("Quit", self)
        self.act_quit.setShortcut("Ctrl+Q")
        self.act_quit.triggered.connect(self.close)

        # Plotting action for CSV tables
        self.act_plot_selected = QAction("Plot Selected Columns", self)
        self.act_plot_selected.setToolTip(
            "Plot selected table columns (first selection is X, others are Y)"
        )
        self.act_plot_selected.triggered.connect(self.plot_selected_columns)
        self.act_plot_selected.setEnabled(False)

    def _create_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        self.addToolBar(tb)
        tb.addAction(self.act_new)
        tb.addAction(self.act_open)
        tb.addAction(self.act_add_files)
        tb.addAction(self.act_add_folder)
        tb.addAction(self.act_new_folder)
        tb.addSeparator()
        tb.addAction(self.act_expand)
        tb.addAction(self.act_collapse)
        tb.addSeparator()
        tb.addAction(self.act_plot_selected)
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

        candidate = "/"
        if kind == "group":
            candidate = path
        elif kind in ("attr", "attrs-folder"):
            candidate = path  # path points to the owner group/dataset for attrs-folder and attr
        elif kind == "dataset":
            # parent group of dataset
            try:
                candidate = posixpath.dirname(path) or "/"
            except Exception:
                candidate = "/"
        elif kind == "file":
            candidate = "/"

        # Safety: if candidate is a CSV-derived group, drop into its parent instead
        try:
            fpath = self.model.filepath
            if fpath and candidate and candidate != "/":
                with h5py.File(fpath, "r") as h5:
                    try:
                        obj = h5[candidate]
                    except Exception:  # noqa: BLE001
                        obj = None
                    if obj is not None and isinstance(obj, h5py.Group):
                        try:
                            is_csv = ('source_type' in obj.attrs and obj.attrs['source_type'] == 'csv')
                        except Exception:  # noqa: BLE001
                            is_csv = False
                        if is_csv:
                            return posixpath.dirname(candidate) or "/"
        except Exception:  # noqa: BLE001
            pass
        return candidate

    def _get_target_group_path_for_index(self, index) -> str:
        if not index or not index.isValid():
            return self._get_target_group_path()
        index = index.sibling(index.row(), 0)
        item = self.model.itemFromIndex(index)
        if item is None:
            return self._get_target_group_path()
        kind = item.data(self.model.ROLE_KIND)
        path = item.data(self.model.ROLE_PATH) or "/"
        # Compute the default candidate target path based on item kind
        if kind == "group":
            candidate = path
        elif kind in ("attr", "attrs-folder"):
            candidate = path  # owner group path
        elif kind == "dataset":
            try:
                candidate = posixpath.dirname(path) or "/"
            except Exception:  # noqa: BLE001
                candidate = "/"
        elif kind == "file":
            candidate = "/"
        else:
            candidate = self._get_target_group_path()

        # If the candidate is a CSV-derived group, redirect the drop to its parent group
        # to avoid placing files inside the CSV group (regardless of expansion state).
        try:
            fpath = self.model.filepath
            if fpath and candidate and candidate != "/" and kind == "group":
                with h5py.File(fpath, "r") as h5:
                    try:
                        obj = h5[candidate]
                    except Exception:  # noqa: BLE001
                        obj = None
                    if obj is not None and isinstance(obj, h5py.Group):
                        try:
                            is_csv = ('source_type' in obj.attrs and obj.attrs['source_type'] == 'csv')
                        except Exception:  # noqa: BLE001
                            is_csv = False
                        if is_csv:
                            parent = posixpath.dirname(candidate) or "/"
                            return parent
        except Exception:  # noqa: BLE001
            pass

        return candidate

    def add_files_dialog(self) -> None:
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.information(self, "No file", "Open an HDF5 file first.")
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Select files to add")
        if not files:
            return
        target_group = self._get_target_group_path()
        added, errors = self._add_items_batch(files, [], target_group)
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
        added, errors = self._add_items_batch([], [directory], target_group)
        self.model.load_file(fpath)
        self.tree.expandToDepth(2)
        if errors:
            QMessageBox.warning(self, "Completed with errors", "\n".join(errors))
        elif added:
            self.statusBar().showMessage(f"Added {added} item(s) under {target_group}", 5000)

    def new_folder_dialog(self) -> None:
        """Create a new empty group (folder) in the HDF5 file."""
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.information(self, "No file", "Open an HDF5 file first.")
            return

        target_group = self._get_target_group_path()

        # Get folder name from user
        from qtpy.QtWidgets import QInputDialog
        folder_name, ok = QInputDialog.getText(
            self,
            "New Folder",
            f"Enter folder name to create in {target_group}:"
        )

        if not ok or not folder_name:
            return

        # Sanitize the folder name
        folder_name = _sanitize_hdf5_name(folder_name)
        if not folder_name:
            QMessageBox.warning(self, "Invalid Name", "Folder name cannot be empty.")
            return

        # Create the full path for the new group
        if target_group == "/":
            new_group_path = "/" + folder_name
        else:
            new_group_path = posixpath.join(target_group, folder_name)

        # Create the group in the HDF5 file
        try:
            with h5py.File(fpath, "r+") as h5:
                if new_group_path in h5:
                    QMessageBox.warning(
                        self,
                        "Already Exists",
                        f"Group '{new_group_path}' already exists."
                    )
                    return
                h5.create_group(new_group_path)

            # Reload the tree and expand to show the new folder
            self.model.load_file(fpath)
            self.tree.expandToDepth(2)
            self.statusBar().showMessage(f"Created folder: {new_group_path}", 5000)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to create folder: {exc}"
            )

    def _move_hdf5_item(self, source_path: str, target_group: str) -> bool:
        """Move an HDF5 dataset or group from source_path to target_group.

        Returns True if successful, False otherwise.
        """
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.warning(self, "No file", "No HDF5 file loaded.")
            return False

        # Can't move to itself
        if source_path == target_group:
            QMessageBox.warning(self, "Invalid Move", "Cannot move item to itself.")
            return False

        # Can't move into its own child
        if target_group.startswith(source_path + "/"):
            QMessageBox.warning(self, "Invalid Move", "Cannot move item into its own child.")
            return False

        try:
            with h5py.File(fpath, "r+") as h5:
                # Check if source exists
                if source_path not in h5:
                    QMessageBox.warning(self, "Not Found", f"Source '{source_path}' not found.")
                    return False

                # Prevent moving items out of CSV groups
                source_parent = posixpath.dirname(source_path) or "/"
                if source_parent and source_parent != "/" and isinstance(h5[source_parent], h5py.Group):
                    parent_grp = h5[source_parent]
                    if 'source_type' in parent_grp.attrs and parent_grp.attrs['source_type'] == 'csv':
                        QMessageBox.warning(
                            self,
                            "Invalid Move",
                            "Cannot move items out of CSV groups."
                        )
                        return False

                # Prevent moving into CSV groups
                if target_group and target_group != "/" and isinstance(h5[target_group], h5py.Group):
                    grp = h5[target_group]
                    if 'source_type' in grp.attrs and grp.attrs['source_type'] == 'csv':
                        QMessageBox.warning(
                            self,
                            "Invalid Target",
                            "Cannot move items into CSV groups."
                        )
                        return False

                # Construct new path
                item_name = os.path.basename(source_path)
                if target_group == "/":
                    new_path = "/" + item_name
                else:
                    new_path = posixpath.join(target_group, item_name)

                # If already at destination, treat as no-op
                if source_path == new_path:
                    return True

                # Check if destination exists
                if new_path in h5:
                    resp = QMessageBox.question(
                        self,
                        "Overwrite?",
                        f"'{new_path}' already exists. Overwrite?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if resp != QMessageBox.Yes:
                        return False
                    del h5[new_path]

                # Perform the move (copy + delete)
                h5.copy(source_path, new_path)
                del h5[source_path]

                return True

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Move Failed",
                f"Failed to move item: {exc}"
            )
            return False

    def _add_items_batch(self, files: list[str], folders: list[str], target_group: str) -> tuple[int, list[str]]:
        fpath = self.model.filepath
        if not fpath:
            return 0, ["No HDF5 file loaded"]
        errors: list[str] = []
        added = 0
        try:
            with h5py.File(fpath, "r+") as h5:
                # Final safety: never allow writing into a CSV-derived group
                try:
                    if target_group and target_group != "/" and isinstance(h5[target_group], h5py.Group):
                        grp = h5[target_group]
                        if 'source_type' in grp.attrs and grp.attrs['source_type'] == 'csv':
                            target_group = posixpath.dirname(target_group) or "/"
                except Exception:  # noqa: BLE001
                    pass

                if target_group == "/":
                    base_grp = h5
                else:
                    base_grp = h5.require_group(target_group)
                for path_on_disk in files:
                    name = os.path.basename(path_on_disk)
                    if name in excluded_files:
                        continue
                    h5_path = posixpath.join(target_group, name) if target_group != "/" else "/" + name
                    try:
                        self._create_dataset_from_file(base_grp, h5_path, path_on_disk, np)
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
                            self._create_dataset_from_file(base_grp, h5_path, path_on_disk, np)
                            added += 1
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{name}: {exc}")
                for directory in folders:
                    base_name = os.path.basename(os.path.normpath(directory))
                    if target_group == "/":
                        root_h5_group = "/" + base_name
                    else:
                        root_h5_group = posixpath.join(target_group, base_name)
                    for dirpath, dirnames, filenames in os.walk(directory):
                        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
                        rel = os.path.relpath(dirpath, directory)
                        rel = "." if rel == "." else rel.replace("\\", "/")
                        current_group_path = (
                            root_h5_group if rel == "." else posixpath.join(root_h5_group, rel)
                        )
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
            return 0, [str(exc)]
        return added, errors

    def _create_dataset_from_file(self, grp, h5_path: str, disk_path: str, np) -> None:
        """Create a dataset at h5_path from a file on disk under the given group (or file root).

        For CSV files, creates a group with individual datasets for each column.
        For other files, stores as text or binary data.

        Raises FileExistsError if the path already exists.
        """
        # Check existence
        f = grp.file
        if h5_path in f:
            raise FileExistsError(h5_path)

        # Special handling for CSV files
        if disk_path.lower().endswith('.csv'):
            self._create_datasets_from_csv(f, h5_path, disk_path)
            return

        # Ensure parent groups exist
        parent = os.path.dirname(h5_path).replace("\\", "/")
        if parent and parent != "/":
            f.require_group(parent)
        # Try text then binary
        try:
            with open(disk_path, "r", encoding="utf-8") as fin:
                data = fin.read()
            # Compress text data with gzip (level 9 for maximum compression)
            compressed = gzip.compress(data.encode('utf-8'), compresslevel=9)
            ds = f.create_dataset(h5_path, data=np.frombuffer(compressed, dtype="uint8"))
            # Mark as compressed text so we can decompress on read
            ds.attrs['compressed'] = 'gzip'
            ds.attrs['original_encoding'] = 'utf-8'
            return
        except Exception:  # noqa: BLE001
            pass
        # Read as binary and compress
        with open(disk_path, "rb") as fin:
            bdata = fin.read()
        # Compress binary data with gzip (level 9 for maximum compression)
        compressed = gzip.compress(bdata, compresslevel=9)
        ds = f.create_dataset(h5_path, data=np.frombuffer(compressed, dtype="uint8"))
        # Mark as compressed binary so we can decompress on read
        ds.attrs['compressed'] = 'gzip'
        ds.attrs['original_encoding'] = 'binary'

    def _create_datasets_from_csv(self, f: h5py.File, h5_path: str, disk_path: str) -> None:
        """Convert a CSV file to HDF5 datasets.

        Creates a group at h5_path (without .csv extension) containing one dataset per column.
        Each dataset contains the column data with appropriate dtype.
        """
        # Create progress dialog
        progress = QProgressDialog("Reading CSV file...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(500)  # Show after 500ms
        progress.setValue(0)
        QApplication.processEvents()

        # Read CSV with pandas
        try:
            df = pd.read_csv(disk_path)
            progress.setValue(20)
            QApplication.processEvents()
            if progress.wasCanceled():
                raise ValueError("CSV import cancelled by user")
        except Exception as exc:  # noqa: BLE001
            progress.close()
            raise ValueError(f"Failed to read CSV file: {exc}") from exc

        # Remove .csv extension from group name
        group_path = h5_path
        if group_path.lower().endswith('.csv'):
            group_path = group_path[:-4]

        # Ensure parent groups exist
        parent = os.path.dirname(group_path).replace("\\", "/")
        if parent and parent != "/":
            f.require_group(parent)

        # Create a group for the CSV data
        grp = f.create_group(group_path)

        # Add metadata about the source file
        grp.attrs['source_file'] = os.path.basename(disk_path)
        grp.attrs['source_type'] = 'csv'
        grp.attrs['column_names'] = list(df.columns)

        progress.setLabelText(f"Creating datasets for {len(df.columns)} columns...")
        progress.setValue(30)
        QApplication.processEvents()

        # Create a dataset for each column
        used_names: set[str] = set()
        column_dataset_names: list[str] = []
        total_cols = len(df.columns)
        for idx, col in enumerate(df.columns):
            if progress.wasCanceled():
                # Clean up partial group
                try:
                    del f[group_path]
                except Exception:  # noqa: BLE001
                    pass
                progress.close()
                raise ValueError("CSV import cancelled by user")

            # Update progress (30-90% range for column processing)
            progress_val = 30 + int((idx / total_cols) * 60)
            progress.setValue(progress_val)
            progress.setLabelText(f"Creating dataset {idx + 1}/{total_cols}: {col}")
            QApplication.processEvents()

            col_data = df[col]

            # Clean column name for use as dataset name
            base = _sanitize_hdf5_name(str(col))
            ds_name = base if base else 'unnamed_column'
            # Ensure uniqueness within the group
            if ds_name in used_names:
                i = 2
                while f"{ds_name}_{i}" in used_names or f"{ds_name}_{i}" in grp:
                    i += 1
                ds_name = f"{ds_name}_{i}"
            used_names.add(ds_name)
            column_dataset_names.append(ds_name)

            # Convert pandas Series to numpy array with appropriate dtype
            if col_data.dtype == 'object':
                # For object dtype, convert to Python list then create dataset
                # This avoids numpy unicode string issues
                try:
                    # Convert to Python strings
                    str_list = [str(x) for x in col_data.values]
                    grp.create_dataset(
                        ds_name,
                        data=str_list,
                        dtype=h5py.string_dtype(encoding='utf-8')
                    )
                except Exception:  # noqa: BLE001
                    # Fallback: convert to bytes
                    str_list = [str(x) for x in col_data.values]
                    grp.create_dataset(
                        ds_name,
                        data=str_list,
                        dtype=h5py.string_dtype(encoding='utf-8')
                    )
            else:
                # Numeric or other numpy-supported dtypes
                grp.create_dataset(ds_name, data=col_data.values)

        # Persist the actual dataset names used for each column (same order as column_names)
        progress.setLabelText("Finalizing CSV import...")
        progress.setValue(95)
        QApplication.processEvents()

        try:
            grp.attrs['column_dataset_names'] = np.array(column_dataset_names, dtype=object)
        except Exception:  # noqa: BLE001
            # Fallback to list assignment if dtype=object attr not permitted
            grp.attrs['column_dataset_names'] = column_dataset_names

        progress.setValue(100)
        progress.close()

    def new_file_dialog(self) -> None:
        """Create a new HDF5 file."""
        last_dir = os.getcwd()
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Create New HDF5 File",
            last_dir,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if not filepath:
            return

        # Add .h5 extension if no extension provided
        if not filepath.endswith(('.h5', '.hdf5')):
            filepath += '.h5'

        # Check if file already exists
        if os.path.exists(filepath):
            resp = QMessageBox.question(
                self,
                "File exists",
                f"File '{filepath}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return

        try:
            # Create a new empty HDF5 file
            with h5py.File(filepath, "w"):
                # Create an empty file with a root group
                pass

            # Load the newly created file
            self.load_hdf5(filepath)
            self.statusBar().showMessage(f"Created new HDF5 file: {filepath}", 5000)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Failed to create file",
                f"Could not create HDF5 file:\n{filepath}\n\n{exc}",
            )

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
        self._set_preview_text("")

    # Selection handling
    def on_selection_changed(self, selected, _deselected) -> None:
        indexes = selected.indexes()
        if not indexes:
            self._hide_attributes()
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
            self.preview_group(path)
        else:
            self.preview_label.setText(str(kind) if kind else "")
            self._set_preview_text("")
            self._hide_attributes()

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

        # Check if this is a CSV group
        is_csv_group = False
        csv_expanded = False
        if kind == "group" and path and self.model.filepath:
            try:
                with h5py.File(self.model.filepath, "r") as h5:
                    grp = h5[path]
                    if isinstance(grp, h5py.Group):
                        if 'source_type' in grp.attrs and grp.attrs['source_type'] == 'csv':
                            is_csv_group = True
                            csv_expanded = item.data(self.model.ROLE_CSV_EXPANDED) or False
            except Exception:  # noqa: BLE001
                pass

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

        # Add CSV group expand/collapse option
        act_toggle_csv = None
        if is_csv_group:
            if csv_expanded:
                act_toggle_csv = menu.addAction("Hide Internal Structure")
            else:
                act_toggle_csv = menu.addAction("Show Internal Structure")
            menu.addSeparator()

        act_delete = None
        if deletable and label:
            act_delete = menu.addAction(label)

        # If no actions available, don't show menu
        if not act_toggle_csv and not act_delete:
            return

        global_pos = self.tree.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)

        if chosen == act_toggle_csv:
            self.model.toggle_csv_group_expansion(item)
        elif chosen == act_delete:
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
            self._set_preview_text("No file loaded")
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
            self._hide_attributes()
            return
        # If the dataset name ends with .png, try to display as image
        if dspath.lower().endswith('.png'):
            try:
                with h5py.File(fpath, "r") as h5:
                    obj = h5[dspath]
                    if not isinstance(obj, h5py.Dataset):
                        self._set_preview_text("Selected path is not a dataset.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        self._hide_attributes()
                        return
                    # Read raw bytes from dataset
                    data = obj[()]

                    # Check if this is compressed binary data
                    if 'compressed' in obj.attrs and obj.attrs['compressed'] == 'gzip':
                        encoding = obj.attrs.get('original_encoding', 'utf-8')
                        if isinstance(encoding, bytes):
                            encoding = encoding.decode('utf-8')
                        if encoding == 'binary' and isinstance(data, np.ndarray) and data.dtype == np.uint8:
                            # Decompress the binary data
                            compressed_bytes = data.tobytes()
                            img_bytes = gzip.decompress(compressed_bytes)
                        elif isinstance(data, bytes):
                            img_bytes = data
                        elif hasattr(data, 'tobytes'):
                            img_bytes = data.tobytes()
                        else:
                            self._set_preview_text("Dataset is not a valid PNG byte array.")
                            self.preview_edit.setVisible(True)
                            self.preview_image.setVisible(False)
                            self._hide_attributes()
                            return
                    elif isinstance(data, bytes):
                        img_bytes = data
                    elif hasattr(data, 'tobytes'):
                        img_bytes = data.tobytes()
                    else:
                        self._set_preview_text("Dataset is not a valid PNG byte array.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        self._hide_attributes()
                        return
                    pixmap = QPixmap()
                    if pixmap.loadFromData(img_bytes, "PNG"):
                        # Scale pixmap to fit preview area, maintaining aspect ratio
                        self._show_scaled_image(pixmap)
                        self.preview_image.setVisible(True)
                        self.preview_edit.setVisible(False)
                        self.preview_table.setVisible(False)
                        self.filter_panel.setVisible(False)
                        # Show attributes for the dataset
                        self._show_attributes(obj)
                    else:
                        self._set_preview_text("Failed to load PNG image from dataset.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        self._hide_attributes()
            except Exception as exc:
                self._set_preview_text(f"Error reading PNG dataset:\n{exc}")
                self.preview_edit.setVisible(True)
                self.preview_image.setVisible(False)
                self._hide_attributes()
            return
        # Otherwise, show text preview for non-PNG datasets
        try:
            with h5py.File(fpath, "r") as h5:
                obj = h5[dspath]
                if not isinstance(obj, h5py.Dataset):
                    self._set_preview_text("Selected path is not a dataset.")
                    self.preview_edit.setVisible(True)
                    self.preview_image.setVisible(False)
                    self._hide_attributes()
                    return
                ds = obj
                text, note = _dataset_to_text(ds, limit_bytes=1_000_000)
                header = f"shape={ds.shape}, dtype={ds.dtype}"
                if note:
                    header += f"\n{note}"

                # Apply syntax highlighting based on file extension
                language = get_language_from_path(dspath)
                self._set_preview_text(header + "\n\n" + text, language=language)
                self.preview_edit.setVisible(True)
                self.preview_image.setVisible(False)
                # Show attributes for the dataset
                self._show_attributes(ds)
        except Exception as exc:
            self._set_preview_text(f"Error reading dataset:\n{exc}")
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
            self._hide_attributes()

    def _set_preview_text(self, text: str, language: str = "plain") -> None:
        """Set preview text with optional syntax highlighting.

        Args:
            text: The text content to display
            language: Language identifier for syntax highlighting (default: "plain")
        """
        # Remove old highlighter if exists
        if self._current_highlighter is not None:
            self._current_highlighter.setDocument(None)
            self._current_highlighter = None

        # Set the text
        self.preview_edit.setPlainText(text)

        # Apply syntax highlighting if not plain text
        if language != "plain":
            try:
                self._current_highlighter = SyntaxHighlighter(
                    self.preview_edit.document(),
                    language=language
                )
            except Exception:  # noqa: BLE001
                # If highlighting fails, just show plain text
                pass

        # Show text view, hide table and image
        self.preview_edit.setVisible(True)
        self.preview_table.setVisible(False)
        self.preview_image.setVisible(False)
        self.filter_panel.setVisible(False)

    def _show_attributes(self, h5_obj) -> None:
        """Display attributes of an HDF5 object in the attributes table.

        Args:
            h5_obj: HDF5 group or dataset object with attributes
        """
        try:
            attrs = dict(h5_obj.attrs)
            if attrs:
                self.attrs_table.setRowCount(len(attrs))
                for row, (key, value) in enumerate(attrs.items()):
                    # Attribute name
                    name_item = QTableWidgetItem(str(key))
                    self.attrs_table.setItem(row, 0, name_item)
                    # Attribute value (convert to string)
                    try:
                        if isinstance(value, (np.ndarray, list, tuple)):
                            # For arrays/lists, show truncated representation
                            value_str = repr(value)
                            if len(value_str) > 200:
                                value_str = value_str[:200] + "..."
                        else:
                            value_str = str(value)
                    except Exception:  # noqa: BLE001
                        value_str = repr(value)
                    value_item = QTableWidgetItem(value_str)
                    self.attrs_table.setItem(row, 1, value_item)
                # Resize columns to content
                self.attrs_table.resizeColumnsToContents()
            else:
                # No attributes, clear the table
                self.attrs_table.setRowCount(0)
        except Exception:  # noqa: BLE001
            # If there's an error, just clear the attributes table
            self.attrs_table.setRowCount(0)

    def _hide_attributes(self) -> None:
        """Hide the attributes table."""
        self.attrs_table.setRowCount(0)

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
            self._set_preview_text("No file loaded")
            self._hide_attributes()
            return
        try:
            with h5py.File(fpath, "r") as h5:
                g = h5[grouppath]
                val = g.attrs[key]
                self._set_preview_text(repr(val))
                self._hide_attributes()
        except Exception as exc:
            self._set_preview_text(f"Error reading attribute:\n{exc}")
            self._hide_attributes()

    def preview_group(self, grouppath: str) -> None:
        """Preview a group. If it's a CSV-derived group, show as table."""
        self.preview_label.setText(f"Group: {grouppath}")
        fpath = self.model.filepath
        if not fpath:
            self._set_preview_text("No file loaded")
            self._hide_attributes()
            return

        try:
            with h5py.File(fpath, "r") as h5:
                grp = h5[grouppath]
                if not isinstance(grp, h5py.Group):
                    self._set_preview_text("(Not a group)")
                    self._hide_attributes()
                    return

                # Check if this is a CSV-derived group
                if 'source_type' in grp.attrs and grp.attrs['source_type'] == 'csv':
                    # Track current CSV group for plotting
                    self._current_csv_group_path = grouppath
                    self._show_csv_table(grp)
                    self._update_plot_action_enabled()
                else:
                    self._current_csv_group_path = None
                    self._set_preview_text("(No content to display)")
                    # Show attributes for the group
                    self._show_attributes(grp)
        except Exception as exc:
            self._set_preview_text(f"Error reading group:\n{exc}")
            self._hide_attributes()

    def _get_th_location(self, ds_key, grp):
        """We allow an optional 'Time History' group for CSV columns."""
        OPTIONAL_GROUP_FOR_COLUMNS = 'Time History'
        th_group = OPTIONAL_GROUP_FOR_COLUMNS in grp
        if th_group:
            key_in_group = ds_key in grp[OPTIONAL_GROUP_FOR_COLUMNS]
            th_grp = grp[OPTIONAL_GROUP_FOR_COLUMNS]
        else:
            key_in_group = ds_key in grp
            th_grp = grp

        return key_in_group, th_grp

    def _show_csv_table(self, grp: h5py.Group) -> None:
        """Display CSV-derived group data in a table widget."""
        progress = None
        try:
            # Get column names (for headers)
            if 'column_names' in grp.attrs:
                try:
                    col_names = [str(c) for c in list(grp.attrs['column_names'])]
                except Exception:  # noqa: BLE001
                    col_names = list(grp.keys())
            else:
                col_names = list(grp.keys())

            # Optional mapping of columns to actual dataset names
            col_ds_names: list[str] | None = None
            if 'column_dataset_names' in grp.attrs:
                try:
                    col_ds_names = [str(c) for c in list(grp.attrs['column_dataset_names'])]
                    if len(col_ds_names) != len(col_names):
                        col_ds_names = None
                except Exception:  # noqa: BLE001
                    col_ds_names = None

            # Estimate total work for progress
            total_cols = len(col_names)

            # Create progress dialog
            progress = QProgressDialog("Loading CSV data...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(500)  # Show after 500ms
            progress.setValue(0)
            QApplication.processEvents()

            # Read all datasets
            data_dict = {}
            max_rows = 0
            for idx, col_name in enumerate(col_names):
                if progress.wasCanceled():
                    progress.close()
                    self._set_preview_text("(CSV display cancelled)")
                    return

                # Update progress (0-70% range for reading data)
                progress_val = int((idx / total_cols) * 70)
                progress.setValue(progress_val)
                progress.setLabelText(f"Reading column {idx + 1}/{total_cols}: {col_name}")
                QApplication.processEvents()

                # Resolve dataset key for this column
                ds_key = None
                if col_ds_names is not None:
                    ds_key = col_ds_names[idx]
                else:
                    # Try sanitized version of the column name
                    cand = _sanitize_hdf5_name(str(col_name))
                    if cand in grp:
                        ds_key = cand
                    elif col_name in grp:
                        ds_key = col_name

                key_in_group, th_grp = self._get_th_location(ds_key, grp)
                if ds_key and key_in_group:
                    ds = th_grp[ds_key]
                    if isinstance(ds, h5py.Dataset):
                        # Read dataset data
                        data = ds[()]
                        if isinstance(data, np.ndarray):
                            data_dict[col_name] = data
                            max_rows = max(max_rows, len(data))
                        else:
                            # Scalar dataset
                            data_dict[col_name] = [data]
                            max_rows = max(max_rows, 1)

            if not data_dict:
                progress.close()
                self._set_preview_text("(No datasets found in CSV group)")
                return

            # Store full data for filtering
            self._csv_data_dict = data_dict
            self._csv_column_names = col_names

            # Limit display to avoid performance issues
            MAX_DISPLAY_ROWS = 10000
            # display_rows = min(max_rows, MAX_DISPLAY_ROWS)
            display_rows = max_rows
            is_truncated = max_rows > MAX_DISPLAY_ROWS

            # Setup table - disable updates for performance
            progress.setLabelText("Setting up table...")
            progress.setValue(75)
            QApplication.processEvents()

            self.preview_table.setUpdatesEnabled(False)
            self.preview_table.setSortingEnabled(False)
            self.preview_table.clear()
            self.preview_table.setRowCount(display_rows)
            self.preview_table.setColumnCount(len(col_names))
            self.preview_table.setHorizontalHeaderLabels(col_names)

            # Populate table
            progress.setLabelText(f"Populating table with {display_rows} rows...")
            progress.setValue(80)
            QApplication.processEvents()

            for col_idx, col_name in enumerate(col_names):
                if progress.wasCanceled():
                    progress.close()
                    self.preview_table.setUpdatesEnabled(True)
                    self._set_preview_text("(CSV display cancelled)")
                    return

                # Update progress (80-95% range for populating table)
                if col_idx % 10 == 0:  # Update every 10 columns to avoid too many updates
                    progress_val = 80 + int((col_idx / len(col_names)) * 15)
                    progress.setValue(progress_val)
                    QApplication.processEvents()

                if col_name in data_dict:
                    col_data = data_dict[col_name]
                    # Convert entire column to strings at once (much faster)
                    if isinstance(col_data, np.ndarray):
                        # Use numpy's vectorized string conversion
                        if col_data.dtype.kind == 'S' or col_data.dtype.kind == 'U':
                            # Byte or unicode strings
                            str_data = [str(v) if not isinstance(v, bytes) else v.decode('utf-8', errors='replace')
                                       for v in col_data[:display_rows]]
                        else:
                            # Numeric or other types - use numpy's string conversion
                            str_data = np.char.mod('%s', col_data[:display_rows])
                    else:
                        str_data = [str(v) for v in col_data[:display_rows]]

                    # Set items in batch
                    for row_idx, value_str in enumerate(str_data):
                        item = QTableWidgetItem(value_str)
                        self.preview_table.setItem(row_idx, col_idx, item)

            # Resize columns to content
            progress.setLabelText("Resizing columns...")
            progress.setValue(95)
            QApplication.processEvents()

            self.preview_table.resizeColumnsToContents()

            # Re-enable updates and sorting
            self.preview_table.setUpdatesEnabled(True)
            #self.preview_table.setSortingEnabled(True)  # don't want sorting since it interfers with selecting columns for plotting.

            # Show table, hide others
            progress.setValue(100)
            self.preview_table.setVisible(True)
            self.preview_edit.setVisible(False)
            self.preview_image.setVisible(False)

            # Show filter panel for CSV tables
            self.filter_panel.setVisible(True)

            # Show attributes for the CSV group
            self._show_attributes(grp)

            # Show message if data was truncated
            if is_truncated:
                self.statusBar().showMessage(
                    f"Displaying first {display_rows:,} of {max_rows:,} rows", 8000
                )

            # Enable/disable plotting action depending on visibility/selection
            self._update_plot_action_enabled()

            progress.close()

            # Load saved filters from HDF5 group
            saved_filters = self._load_filters_from_hdf5(grp)
            if saved_filters:
                self._csv_filters = saved_filters
                self.statusBar().showMessage(
                    f"Loaded {len(saved_filters)} saved filter(s) from HDF5 file", 5000
                )

            # Apply any existing filters
            if self._csv_filters:
                self._apply_filters()
            else:
                # No filters - all rows are visible
                self.filter_status_label.setText("No filters applied")
                self.btn_clear_filters.setEnabled(False)
                self._csv_filtered_indices = np.arange(max_rows)
                # Notify model that no filtering is active
                if self._current_csv_group_path and self.model:
                    self.model.set_csv_filtered_indices(self._current_csv_group_path, None)

        except Exception as exc:
            if progress:
                progress.close()
            self._set_preview_text(f"Error displaying CSV table:\n{exc}")
            self.preview_table.setVisible(False)
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
            self._hide_attributes()

    def _get_selected_column_indices(self) -> list[int]:
        try:
            sel_model = self.preview_table.selectionModel()
            if not sel_model:
                return []
            # Prefer selectedColumns if available
            cols = []
            try:
                cols = [idx.column() for idx in sel_model.selectedColumns()]
            except Exception:  # noqa: BLE001
                # Fallback: derive from selectedIndexes
                cols = list({idx.column() for idx in sel_model.selectedIndexes()})
            # Unique and sorted for stable behavior
            return sorted({c for c in cols if c >= 0})
        except Exception:  # noqa: BLE001
            return []

    def _update_plot_action_enabled(self) -> None:
        # Enable plotting when a CSV group is active and >= 2 columns are selected
        is_csv = self._current_csv_group_path is not None and self.preview_table.isVisible()
        sel_cols = self._get_selected_column_indices() if is_csv else []
        self.act_plot_selected.setEnabled(is_csv and len(sel_cols) >= 2)

    def _read_csv_columns(self, group_path: str, column_names: list[str]) -> dict[str, np.ndarray]:
        """Read one or more column arrays from a CSV-derived group by original column names.

        Returns a dict mapping original column name -> numpy array (1-D). Strings remain strings.
        """
        result: dict[str, np.ndarray] = {}
        fpath = self.model.filepath
        if not fpath:
            return result
        try:
            with h5py.File(fpath, "r") as h5:
                grp = h5[group_path]
                # Resolve mapping from original names to dataset keys if present
                mapping: dict[str, str] = {}
                if 'column_names' in grp.attrs:
                    try:
                        orig = [str(c) for c in list(grp.attrs['column_names'])]
                    except Exception:  # noqa: BLE001
                        orig = []
                    ds_names: list[str] | None = None
                    if 'column_dataset_names' in grp.attrs:
                        try:
                            ds_names = [str(c) for c in list(grp.attrs['column_dataset_names'])]
                            if len(ds_names) != len(orig):
                                ds_names = None
                        except Exception:  # noqa: BLE001
                            ds_names = None
                    for i, name in enumerate(orig):
                        key = None
                        if ds_names is not None:
                            key = ds_names[i]
                        else:
                            cand = _sanitize_hdf5_name(name)
                            key = cand if cand in grp else (name if name in grp else None)
                        if key is not None:
                            mapping[name] = key
                # Read requested columns
                for name in column_names:
                    ds_key = mapping.get(name)
                    if ds_key is None:
                        # Fallback to direct/sanitized key lookup
                        cand = _sanitize_hdf5_name(name)
                        if cand in grp:
                            ds_key = cand
                        elif name in grp:
                            ds_key = name
                    key_in_group, th_grp = self._get_th_location(ds_key, grp)
                    if ds_key is None or not key_in_group:
                        continue
                    ds = th_grp[ds_key]
                    if not isinstance(ds, h5py.Dataset):
                        continue
                    data = ds[()]
                    if isinstance(data, np.ndarray):
                        arr = data
                    else:
                        arr = np.array([data])
                    result[name] = arr
        except Exception:  # noqa: BLE001
            return result
        return result

    def plot_selected_columns(self) -> None:
        """Plot selected columns from the current CSV table using matplotlib.

        - First selected (or current) column is X
        - Subsequent selected columns are Y series
        - Adds legend and shows plot in embedded canvas
        """
        if self._current_csv_group_path is None or not self.preview_table.isVisible():
            QMessageBox.information(self, "Plot", "No CSV table is active to plot.")
            return

        # Check if matplotlib is available
        if not self._matplotlib_available:
            QMessageBox.critical(
                self,
                "Matplotlib not available",
                "Matplotlib is not available. Please install it to use plotting features.",
            )
            return

        # Determine selected columns and their order
        sel_cols = self._get_selected_column_indices()
        if len(sel_cols) < 2:
            QMessageBox.information(
                self,
                "Plot",
                "Select at least two columns (first is X, others are Y).",
            )
            return
        # Use current column as X if part of selection; else use the leftmost selected
        current_col = self.preview_table.currentColumn()
        x_idx = current_col if current_col in sel_cols else min(sel_cols)
        y_idxs = [c for c in sel_cols if c != x_idx]

        # Column names from headers
        headers = [
            self.preview_table.horizontalHeaderItem(i).text()
            if self.preview_table.horizontalHeaderItem(i) is not None else f"col_{i}"
            for i in range(self.preview_table.columnCount())
        ]
        try:
            x_name = headers[x_idx]
            y_names = [headers[i] for i in y_idxs]
        except Exception:
            QMessageBox.warning(self, "Plot", "Failed to resolve column headers for plotting.")
            return

        # Use filtered data from the table instead of reading from HDF5
        # This ensures we only plot what's visible (respecting filters)
        if not self._csv_data_dict or self._csv_filtered_indices is None:
            QMessageBox.warning(self, "Plot", "No CSV data available.")
            return

        col_data = {}
        for name in [x_name] + y_names:
            if name in self._csv_data_dict:
                # Get only the filtered rows
                full_data = self._csv_data_dict[name]
                if isinstance(full_data, np.ndarray):
                    col_data[name] = full_data[self._csv_filtered_indices]
                else:
                    col_data[name] = np.array([full_data[i] for i in self._csv_filtered_indices])

        if x_name not in col_data or not any(name in col_data for name in y_names):
            QMessageBox.warning(self, "Plot", "Failed to get selected columns for plotting.")
            return

        # Prepare numeric data, align lengths
        try:
            # Coerce X to numeric
            import pandas as _pd  # local alias
            x_arr = col_data[x_name]
            # Ensure 1-D
            x_arr = x_arr.ravel()
            min_len = min(len(x_arr), *(len(col_data.get(n, [])) for n in y_names if n in col_data))
            if min_len <= 0:
                QMessageBox.warning(self, "Plot", "No data to plot.")
                return
            x_num = _pd.to_numeric(_pd.Series(x_arr[:min_len]), errors="coerce").astype(float).to_numpy()

            # Clear the previous plot
            self.plot_figure.clear()

            # Create a subplot
            ax = self.plot_figure.add_subplot(111)

            any_plotted = False
            for y_name in y_names:
                if y_name not in col_data:
                    continue
                y_arr = col_data[y_name].ravel()[:min_len]
                y_num = _pd.to_numeric(_pd.Series(y_arr), errors="coerce").astype(float).to_numpy()
                import numpy as _np
                valid = _np.isfinite(x_num) & _np.isfinite(y_num)
                if valid.any():
                    ax.plot(x_num[valid], y_num[valid], label=y_name)
                    any_plotted = True

            if not any_plotted:
                QMessageBox.information(self, "Plot", "No valid numeric data found to plot.")
                return

            ax.set_xlabel(x_name)
            ax.set_ylabel(", ".join(y_names))

            try:
                # Use group base name as title
                title = os.path.basename(self._current_csv_group_path.rstrip("/"))
                # Add filter indicator if filters are active
                if self._csv_filters:
                    total_rows = max(len(self._csv_data_dict[col]) for col in self._csv_data_dict)
                    filtered_rows = len(self._csv_filtered_indices) if self._csv_filtered_indices is not None else 0
                    title += f" ({filtered_rows}/{total_rows} rows, filtered)"
                ax.set_title(title)
            except Exception:
                pass

            ax.legend()
            self.plot_figure.tight_layout()

            # Refresh the canvas to display the plot
            self.plot_canvas.draw()

            # Switch to the Plot tab
            self.bottom_tabs.setCurrentIndex(1)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Plot error", f"Failed to plot data:\n{exc}")

    def _configure_filters_dialog(self):
        """Open dialog to configure column filters."""
        if not self._csv_column_names:
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        dialog = ColumnFilterDialog(self._csv_column_names, self)
        dialog.set_filters(self._csv_filters)

        if dialog.exec() == QDialog.Accepted:
            self._csv_filters = dialog.get_filters()
            self._save_filters_to_hdf5()
            self._apply_filters()

    def _clear_filters(self):
        """Clear all active filters and show full dataset."""
        self._csv_filters = []
        self._save_filters_to_hdf5()
        self._apply_filters()

    def _save_filters_to_hdf5(self):
        """Save current filters to the HDF5 file as a JSON attribute."""
        if not self._current_csv_group_path or not self.model or not self.model.filepath:
            return

        try:
            import json
            with h5py.File(self.model.filepath, "r+") as h5:
                if self._current_csv_group_path in h5:
                    grp = h5[self._current_csv_group_path]
                    if isinstance(grp, h5py.Group):
                        if self._csv_filters:
                            # Convert filters to JSON string
                            # Format: list of [column_name, operator, value]
                            filters_json = json.dumps(self._csv_filters)
                            grp.attrs['csv_filters'] = filters_json
                            self.statusBar().showMessage(
                                f"Saved {len(self._csv_filters)} filter(s) to HDF5 file", 3000
                            )
                        else:
                            # Remove filter attribute if no filters
                            if 'csv_filters' in grp.attrs:
                                del grp.attrs['csv_filters']
                                self.statusBar().showMessage("Cleared filters from HDF5 file", 3000)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Warning: Could not save filters: {exc}", 5000)

    def _load_filters_from_hdf5(self, grp: h5py.Group) -> list:
        """Load filters from the HDF5 group attributes.

        Args:
            grp: HDF5 group to load filters from

        Returns:
            List of filters in format [column_name, operator, value]
        """
        try:
            if 'csv_filters' in grp.attrs:
                import json
                filters_json = grp.attrs['csv_filters']
                if isinstance(filters_json, bytes):
                    filters_json = filters_json.decode('utf-8')
                filters = json.loads(filters_json)
                # Validate format
                if isinstance(filters, list):
                    # Ensure each filter is a 3-element list
                    valid_filters = []
                    for f in filters:
                        if isinstance(f, list) and len(f) == 3:
                            valid_filters.append(f)
                    return valid_filters
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not load filters from HDF5: {exc}")
        return []

    def _apply_filters(self):
        """Apply current filters to the CSV table."""
        if not self._csv_data_dict:
            return

        # Update filter status label
        if self._csv_filters:
            filter_text = f"{len(self._csv_filters)} filter(s) applied"
            self.filter_status_label.setText(filter_text)
            self.btn_clear_filters.setEnabled(True)
        else:
            self.filter_status_label.setText("No filters applied")
            self.btn_clear_filters.setEnabled(False)

        # Determine which rows pass all filters
        max_rows = max(len(self._csv_data_dict[col]) for col in self._csv_data_dict)
        valid_rows = np.ones(max_rows, dtype=bool)

        for col_name, operator, value_str in self._csv_filters:
            if col_name not in self._csv_data_dict:
                continue

            col_data = self._csv_data_dict[col_name]
            col_mask = self._evaluate_filter(col_data, operator, value_str)

            # Ensure mask is same length as valid_rows
            if len(col_mask) != len(valid_rows):
                col_mask = np.resize(col_mask, len(valid_rows))

            valid_rows &= col_mask

        # Get indices of valid rows
        filtered_indices = np.where(valid_rows)[0]

        # Store filtered indices for plotting
        self._csv_filtered_indices = filtered_indices

        # Notify the model about filtered indices for CSV export
        if self._current_csv_group_path and self.model:
            if len(filtered_indices) == max_rows:
                # No filtering active, clear stored indices
                self.model.set_csv_filtered_indices(self._current_csv_group_path, None)
            else:
                # Set filtered indices
                self.model.set_csv_filtered_indices(self._current_csv_group_path, filtered_indices)

        # Update table with filtered data
        self.preview_table.setUpdatesEnabled(False)
        self.preview_table.setRowCount(len(filtered_indices))

        for col_idx, col_name in enumerate(self._csv_column_names):
            if col_name not in self._csv_data_dict:
                continue

            col_data = self._csv_data_dict[col_name]

            # Get filtered data
            if isinstance(col_data, np.ndarray):
                filtered_data = col_data[filtered_indices]
                # Convert to strings
                if filtered_data.dtype.kind == 'S' or filtered_data.dtype.kind == 'U':
                    str_data = [str(v) if not isinstance(v, bytes) else v.decode('utf-8', errors='replace')
                               for v in filtered_data]
                else:
                    str_data = np.char.mod('%s', filtered_data)
            else:
                filtered_data = [col_data[i] for i in filtered_indices]
                str_data = [str(v) for v in filtered_data]

            # Set items
            for row_idx, value_str in enumerate(str_data):
                item = QTableWidgetItem(value_str)
                self.preview_table.setItem(row_idx, col_idx, item)

        self.preview_table.setUpdatesEnabled(True)

        # Update status message
        if self._csv_filters:
            total_rows = max_rows
            shown_rows = len(filtered_indices)
            self.statusBar().showMessage(
                f"Showing {shown_rows:,} of {total_rows:,} rows (filtered)", 5000
            )

    def _evaluate_filter(self, col_data, operator, value_str):
        """Evaluate a filter condition on column data.

        Returns a boolean mask of the same length as col_data.
        """
        try:
            # Try numeric comparison first
            if operator in ["==", "!=", ">", ">=", "<", "<="]:
                try:
                    # Try to convert to numeric
                    value_num = float(value_str)
                    if isinstance(col_data, np.ndarray):
                        # Try numeric comparison
                        col_numeric = pd.to_numeric(pd.Series(col_data), errors='coerce')
                        if operator == "==":
                            return col_numeric == value_num
                        elif operator == "!=":
                            return col_numeric != value_num
                        elif operator == ">":
                            return col_numeric > value_num
                        elif operator == ">=":
                            return col_numeric >= value_num
                        elif operator == "<":
                            return col_numeric < value_num
                        elif operator == "<=":
                            return col_numeric <= value_num
                except (ValueError, TypeError):
                    # Fall back to string comparison
                    pass

            # String-based operations
            if isinstance(col_data, np.ndarray):
                # Convert to string array for comparison
                str_array = np.array([str(v) if not isinstance(v, bytes)
                                     else v.decode('utf-8', errors='replace')
                                     for v in col_data])
            else:
                str_array = np.array([str(v) for v in col_data])

            if operator == "==":
                return str_array == value_str
            elif operator == "!=":
                return str_array != value_str
            elif operator == "contains":
                return np.array([value_str in s for s in str_array])
            elif operator == "startswith":
                return np.array([s.startswith(value_str) for s in str_array])
            elif operator == "endswith":
                return np.array([s.endswith(value_str) for s in str_array])
            else:
                # Default: no filter
                return np.ones(len(col_data), dtype=bool)

        except Exception:  # noqa: BLE001
            # On error, don't filter any rows
            return np.ones(len(col_data), dtype=bool)


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
