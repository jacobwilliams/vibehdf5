from __future__ import annotations

import binascii
import fnmatch
import gzip
import json
import os
import posixpath
import sys
import tempfile
import time
import traceback
from pathlib import Path

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure
from qtpy.QtCore import QMimeData, QUrl, Qt
from qtpy.QtGui import QAction, QColor, QDoubleValidator, QDrag, QFont, QFontDatabase, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .hdf5_tree_model import HDF5TreeModel
from .syntax_highlighter import SyntaxHighlighter, get_language_from_path
from .utilities import excluded_dirs, excluded_files


class DraggablePlotListWidget(QListWidget):
    """QListWidget that supports drag-and-drop to export plots to filesystem."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.parent_viewer = parent

    def mimeData(self, items):
        """Create mime data for drag operation."""
        mime_data = super().mimeData(items)
        if items and self.parent_viewer:
            # Store the row index in the mime data
            row = self.row(items[0])
            mime_data.setText(str(row))
        return mime_data

    def startDrag(self, supportedActions):
        """Start drag operation and export plot to temporary file."""
        current_row = self.currentRow()
        if current_row < 0 or not self.parent_viewer:
            return

        # Export the plot to a temporary file
        try:
            # Get plot configuration
            plot_config = self.parent_viewer._saved_plots[current_row]
            plot_name = plot_config.get("name", "plot")
            export_format = plot_config.get("plot_options", {}).get("export_format", "png")

            # Sanitize filename
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in plot_name)
            filename = f"{safe_name}.{export_format}"

            # Create temporary file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, filename)

            # Export the plot
            success, error_msg = self.parent_viewer._export_plot_to_file(plot_config, temp_path)

            if success:
                # Create drag with file URL
                drag = QDrag(self)
                mime_data = QMimeData()
                mime_data.setUrls([QUrl.fromLocalFile(temp_path)])
                mime_data.setText(filename)
                drag.setMimeData(mime_data)

                # Execute drag operation
                drag.exec_(Qt.CopyAction)
            else:
                QMessageBox.warning(self, "Export Failed", f"Failed to export plot for drag-and-drop.\n\nError: {error_msg}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.warning(self, "Export Error", f"Error exporting plot: {e}\n\n{tb}")


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
        if "compressed" in ds.attrs and ds.attrs["compressed"] == "gzip":
            if isinstance(data, np.ndarray) and data.dtype == np.uint8:
                compressed_bytes = data.tobytes()
                decompressed = gzip.decompress(compressed_bytes)
                encoding = ds.attrs.get("original_encoding", "utf-8")
                if isinstance(encoding, bytes):
                    encoding = encoding.decode("utf-8")
                # Check if this is binary data
                if encoding == "binary":
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
    if isinstance(data, np.ndarray) and data.dtype.kind == "S":
        try:
            # Flatten and join bytes chunks
            b = b"".join(x.tobytes() if hasattr(x, "tobytes") else bytes(x) for x in data.ravel())
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
                text = "\n".join(map(str, as_str.ravel().tolist()))
            else:
                text = str(as_str)
            note = None
            if len(text.encode("utf-8")) > limit_bytes:
                enc = text.encode("utf-8")[:limit_bytes]
                text = enc.decode("utf-8", errors="ignore") + "\n… (truncated)"
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
        t = t[:200_000] + "… (truncated)"
        note = "Preview truncated"
    return t, note


def _bytes_to_text(
    b: bytes, limit_bytes: int = 1_000_000, decompressed: bool = False
) -> tuple[str, str | None]:
    note = None
    if len(b) > limit_bytes:
        b = b[:limit_bytes]
        note = f"Preview limited to {limit_bytes} bytes"
        if decompressed:
            note = f"Preview limited to {limit_bytes} bytes (decompressed)"
    elif decompressed:
        note = "(decompressed from gzip)"
    try:
        return b.decode("utf-8"), note
    except UnicodeDecodeError:
        # Provide a hex dump preview
        hexstr = binascii.hexlify(b).decode("ascii")
        # Group hex bytes in pairs for readability
        grouped = " ".join(hexstr[i : i + 2] for i in range(0, len(hexstr), 2))
        if len(grouped) > 200_000:
            grouped = grouped[:200_000] + "… (truncated)"
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
            source_path = bytes(md.data("application/x-hdf5-path")).decode("utf-8")
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


class ColumnStatisticsDialog(QDialog):
    """Dialog for displaying column statistics."""

    def __init__(self, column_names, data_dict, filtered_indices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Column Statistics")
        self.resize(700, 500)

        self.column_names = column_names
        self.data_dict = data_dict
        self.filtered_indices = filtered_indices

        layout = QVBoxLayout(self)

        # Info label
        if filtered_indices is not None and len(filtered_indices) > 0:
            total_rows = max(len(data_dict[col]) for col in data_dict if col in column_names)
            info_text = f"Statistics for {len(filtered_indices)} filtered rows (out of {total_rows} total)"
        else:
            total_rows = max(len(data_dict[col]) for col in data_dict if col in column_names)
            info_text = f"Statistics for all {total_rows} rows"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(info_label)

        # Statistics table
        self.stats_table = QTableWidget(self)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.stats_table)

        # Calculate and display statistics
        self._calculate_statistics()

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _calculate_statistics(self):
        """Calculate statistics for all columns."""
        # Define statistics to calculate
        stat_labels = ["Count", "Min", "Max", "Mean", "Median", "Std Dev", "Sum", "Unique Values"]

        # Set up table dimensions
        self.stats_table.setRowCount(len(stat_labels))
        self.stats_table.setColumnCount(len(self.column_names))
        self.stats_table.setHorizontalHeaderLabels(self.column_names)
        self.stats_table.setVerticalHeaderLabels(stat_labels)

        # Calculate statistics for each column
        for col_idx, col_name in enumerate(self.column_names):
            if col_name not in self.data_dict:
                continue

            col_data = self.data_dict[col_name]

            # Apply filtering if needed
            if self.filtered_indices is not None and len(self.filtered_indices) > 0:
                if isinstance(col_data, np.ndarray):
                    filtered_data = col_data[self.filtered_indices]
                else:
                    filtered_data = [col_data[i] for i in self.filtered_indices]
            else:
                filtered_data = col_data

            # Convert to pandas Series for easier statistics
            try:
                if isinstance(filtered_data, np.ndarray):
                    series = pd.Series(filtered_data)
                else:
                    series = pd.Series(list(filtered_data))

                # Try to convert to numeric
                numeric_series = pd.to_numeric(series, errors="coerce")
                is_numeric = not numeric_series.isna().all()

                # Calculate statistics
                stats = {}
                stats["Count"] = len(series.dropna())

                if is_numeric:
                    # Numeric statistics
                    stats["Min"] = f"{numeric_series.min():.6g}" if not numeric_series.isna().all() else "N/A"
                    stats["Max"] = f"{numeric_series.max():.6g}" if not numeric_series.isna().all() else "N/A"
                    stats["Mean"] = f"{numeric_series.mean():.6g}" if not numeric_series.isna().all() else "N/A"
                    stats["Median"] = f"{numeric_series.median():.6g}" if not numeric_series.isna().all() else "N/A"
                    stats["Std Dev"] = f"{numeric_series.std():.6g}" if not numeric_series.isna().all() else "N/A"
                    stats["Sum"] = f"{numeric_series.sum():.6g}" if not numeric_series.isna().all() else "N/A"
                else:
                    # String statistics
                    stats["Min"] = str(series.min()) if len(series) > 0 else "N/A"
                    stats["Max"] = str(series.max()) if len(series) > 0 else "N/A"
                    stats["Mean"] = "N/A"
                    stats["Median"] = "N/A"
                    stats["Std Dev"] = "N/A"
                    stats["Sum"] = "N/A"

                stats["Unique Values"] = str(series.nunique())

                # Populate table
                for row_idx, stat_label in enumerate(stat_labels):
                    value = stats.get(stat_label, "N/A")
                    item = QTableWidgetItem(str(value))
                    self.stats_table.setItem(row_idx, col_idx, item)

            except Exception as e:
                # On error, fill with N/A
                for row_idx in range(len(stat_labels)):
                    item = QTableWidgetItem("Error")
                    item.setToolTip(str(e))
                    self.stats_table.setItem(row_idx, col_idx, item)

        # Resize columns to content
        self.stats_table.resizeColumnsToContents()


class ColumnSortDialog(QDialog):
    """Dialog for configuring multi-column sorting."""

    def __init__(self, column_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Column Sorting")
        self.resize(500, 400)

        self.column_names = column_names
        self.sort_specs = []  # List of (column_name, ascending) tuples

        layout = QVBoxLayout(self)

        # Instructions
        info_label = QLabel(
            "Add columns to sort by. Rows will be sorted by the first column, "
            "then by the second column (for equal values), and so on."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Scroll area for sort specifications
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.StyledPanel)

        sort_container = QWidget()
        self.sort_layout = QVBoxLayout(sort_container)
        self.sort_layout.setContentsMargins(5, 5, 5, 5)
        self.sort_layout.addStretch()

        scroll.setWidget(sort_container)
        layout.addWidget(scroll)

        # Add sort button
        add_btn = QPushButton("+ Add Sort Column")
        add_btn.clicked.connect(self._add_sort_row)
        layout.addWidget(add_btn)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _add_sort_row(self, col_name=None, ascending=True):
        """Add a new sort row to the dialog."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        # Column dropdown
        col_combo = QComboBox()
        col_combo.addItems(self.column_names)
        if col_name and col_name in self.column_names:
            col_combo.setCurrentText(col_name)
        col_combo.setMinimumWidth(200)

        # Order dropdown
        order_combo = QComboBox()
        order_combo.addItems(["Ascending", "Descending"])
        order_combo.setCurrentText("Ascending" if ascending else "Descending")
        order_combo.setMinimumWidth(120)

        # Move up button
        up_btn = QPushButton("↑")
        up_btn.setMaximumWidth(30)
        up_btn.clicked.connect(lambda: self._move_sort_row(row_widget, -1))

        # Move down button
        down_btn = QPushButton("↓")
        down_btn.setMaximumWidth(30)
        down_btn.clicked.connect(lambda: self._move_sort_row(row_widget, 1))

        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_sort_row(row_widget))

        row_layout.addWidget(QLabel("Column:"))
        row_layout.addWidget(col_combo)
        row_layout.addWidget(QLabel("Order:"))
        row_layout.addWidget(order_combo)
        row_layout.addWidget(up_btn)
        row_layout.addWidget(down_btn)
        row_layout.addWidget(remove_btn)
        row_layout.addStretch()

        # Store references for later retrieval
        row_widget._col_combo = col_combo
        row_widget._order_combo = order_combo

        # Insert before the stretch
        self.sort_layout.insertWidget(self.sort_layout.count() - 1, row_widget)

    def _move_sort_row(self, row_widget, direction):
        """Move a sort row up or down in the list."""
        current_index = None
        for i in range(self.sort_layout.count() - 1):  # -1 to skip stretch
            if self.sort_layout.itemAt(i).widget() == row_widget:
                current_index = i
                break

        if current_index is None:
            return

        new_index = current_index + direction
        # Check bounds (can't move past first or last position before stretch)
        if new_index < 0 or new_index >= self.sort_layout.count() - 1:
            return

        # Remove and re-insert at new position
        self.sort_layout.removeWidget(row_widget)
        self.sort_layout.insertWidget(new_index, row_widget)

    def _remove_sort_row(self, row_widget):
        """Remove a sort row from the dialog."""
        self.sort_layout.removeWidget(row_widget)
        row_widget.deleteLater()

    def get_sort_specs(self):
        """Return list of sort specifications as (column_name, ascending) tuples."""
        sort_specs = []
        for i in range(self.sort_layout.count() - 1):  # -1 to skip stretch
            widget = self.sort_layout.itemAt(i).widget()
            if widget and hasattr(widget, "_col_combo"):
                col_name = widget._col_combo.currentText()
                ascending = widget._order_combo.currentText() == "Ascending"
                sort_specs.append((col_name, ascending))
        return sort_specs

    def set_sort_specs(self, sort_specs):
        """Set the sort specifications to display in the dialog."""
        # Clear existing rows
        for i in reversed(range(self.sort_layout.count() - 1)):  # -1 to skip stretch
            widget = self.sort_layout.itemAt(i).widget()
            if widget:
                self.sort_layout.removeWidget(widget)
                widget.deleteLater()

        # Add rows for each sort spec
        for col_name, ascending in sort_specs:
            self._add_sort_row(col_name, ascending)


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
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
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
            if widget and hasattr(widget, "_col_combo"):
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


class PlotOptionsDialog(QDialog):
    """Dialog for configuring plot options (title, labels, line styles, etc.)."""

    # Available line styles and colors
    LINE_STYLES = ["-", "--", "-.", ":", "None"]
    LINE_STYLE_NAMES = ["Solid", "Dashed", "Dash-dot", "Dotted", "None"]

    # Get matplotlib's default color cycle
    try:
        _prop_cycle = plt.rcParams["axes.prop_cycle"]
        COLORS = _prop_cycle.by_key()["color"]
    except Exception:
        # Fallback to matplotlib's default tab10 colors if prop_cycle not available
        COLORS = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    MARKERS = ["", "o", "s", "^", "v", "D", "*", "+", "x", "."]
    MARKER_NAMES = [
        "None",
        "Circle",
        "Square",
        "Triangle Up",
        "Triangle Down",
        "Diamond",
        "Star",
        "Plus",
        "X",
        "Point",
    ]

    def __init__(self, plot_config, column_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Options")
        self.resize(600, 500)

        self.plot_config = plot_config.copy()  # Work on a copy
        self.column_names = column_names

        layout = QVBoxLayout(self)

        # Create tab widget for different option categories
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: General options
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        # Plot name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Plot Name:"))
        self.name_edit = QLineEdit(self.plot_config.get("name", "Plot"))
        name_layout.addWidget(self.name_edit)
        general_layout.addLayout(name_layout)

        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Plot Title:"))
        self.title_edit = QLineEdit(self.plot_config.get("plot_options", {}).get("title", ""))
        self.title_edit.setPlaceholderText("Auto-generated from CSV group name")
        title_layout.addWidget(self.title_edit)
        general_layout.addLayout(title_layout)

        # X-axis label
        xlabel_layout = QHBoxLayout()
        xlabel_layout.addWidget(QLabel("X-axis Label:"))
        self.xlabel_edit = QLineEdit(self.plot_config.get("plot_options", {}).get("xlabel", ""))
        self.xlabel_edit.setPlaceholderText("Auto-generated from column name")
        xlabel_layout.addWidget(self.xlabel_edit)
        general_layout.addLayout(xlabel_layout)

        # Y-axis label
        ylabel_layout = QHBoxLayout()
        ylabel_layout.addWidget(QLabel("Y-axis Label:"))
        self.ylabel_edit = QLineEdit(self.plot_config.get("plot_options", {}).get("ylabel", ""))
        self.ylabel_edit.setPlaceholderText("Auto-generated from column names")
        ylabel_layout.addWidget(self.ylabel_edit)
        general_layout.addLayout(ylabel_layout)

        # Grid options
        grid_group = QWidget()
        grid_layout = QHBoxLayout(grid_group)
        grid_layout.setContentsMargins(0, 10, 0, 10)

        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(self.plot_config.get("plot_options", {}).get("grid", True))
        grid_layout.addWidget(self.grid_checkbox)

        self.legend_checkbox = QCheckBox("Show Legend")
        self.legend_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("legend", True)
        )
        grid_layout.addWidget(self.legend_checkbox)

        grid_layout.addStretch()
        general_layout.addWidget(grid_group)

        # Axis limits section
        limits_label = QLabel("<b>Axis Limits:</b>")
        general_layout.addWidget(limits_label)

        # X-axis limits
        xlim_layout = QHBoxLayout()
        xlim_layout.addWidget(QLabel("X-axis:"))
        xlim_layout.addWidget(QLabel("Min:"))
        self.xlim_min_edit = QLineEdit()
        self.xlim_min_edit.setPlaceholderText("auto")
        self.xlim_min_edit.setMaximumWidth(100)
        xlim_min_val = self.plot_config.get("plot_options", {}).get("xlim_min", "")
        if xlim_min_val not in (None, ""):
            self.xlim_min_edit.setText(str(xlim_min_val))
        xlim_layout.addWidget(self.xlim_min_edit)

        xlim_layout.addWidget(QLabel("Max:"))
        self.xlim_max_edit = QLineEdit()
        self.xlim_max_edit.setPlaceholderText("auto")
        self.xlim_max_edit.setMaximumWidth(100)
        xlim_max_val = self.plot_config.get("plot_options", {}).get("xlim_max", "")
        if xlim_max_val not in (None, ""):
            self.xlim_max_edit.setText(str(xlim_max_val))
        xlim_layout.addWidget(self.xlim_max_edit)

        xlim_layout.addStretch()
        general_layout.addLayout(xlim_layout)

        # Y-axis limits
        ylim_layout = QHBoxLayout()
        ylim_layout.addWidget(QLabel("Y-axis:"))
        ylim_layout.addWidget(QLabel("Min:"))
        self.ylim_min_edit = QLineEdit()
        self.ylim_min_edit.setPlaceholderText("auto")
        self.ylim_min_edit.setMaximumWidth(100)
        ylim_min_val = self.plot_config.get("plot_options", {}).get("ylim_min", "")
        if ylim_min_val not in (None, ""):
            self.ylim_min_edit.setText(str(ylim_min_val))
        ylim_layout.addWidget(self.ylim_min_edit)

        ylim_layout.addWidget(QLabel("Max:"))
        self.ylim_max_edit = QLineEdit()
        self.ylim_max_edit.setPlaceholderText("auto")
        self.ylim_max_edit.setMaximumWidth(100)
        ylim_max_val = self.plot_config.get("plot_options", {}).get("ylim_max", "")
        if ylim_max_val not in (None, ""):
            self.ylim_max_edit.setText(str(ylim_max_val))
        ylim_layout.addWidget(self.ylim_max_edit)

        ylim_layout.addStretch()
        general_layout.addLayout(ylim_layout)

        # Log scale options
        log_scale_label = QLabel("<b>Logarithmic Scale:</b>")
        general_layout.addWidget(log_scale_label)

        log_scale_layout = QHBoxLayout()
        log_scale_layout.setContentsMargins(0, 5, 0, 10)

        self.xlog_checkbox = QCheckBox("X-axis Log Scale")
        self.xlog_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("xlog", False)
        )
        log_scale_layout.addWidget(self.xlog_checkbox)

        self.ylog_checkbox = QCheckBox("Y-axis Log Scale")
        self.ylog_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("ylog", False)
        )
        log_scale_layout.addWidget(self.ylog_checkbox)

        log_scale_layout.addStretch()
        general_layout.addWidget(QWidget())  # Spacer
        general_layout.addLayout(log_scale_layout)

        # Date/Time X-axis options
        datetime_label = QLabel("<b>Date/Time X-axis:</b>")
        general_layout.addWidget(datetime_label)

        datetime_layout = QVBoxLayout()
        datetime_layout.setContentsMargins(0, 5, 0, 10)

        self.xaxis_datetime_checkbox = QCheckBox("X-axis is Date/Time")
        self.xaxis_datetime_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("xaxis_datetime", False)
        )
        datetime_layout.addWidget(self.xaxis_datetime_checkbox)

        # Date format input
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Date Format:"))
        self.datetime_format_edit = QLineEdit()
        self.datetime_format_edit.setPlaceholderText("%Y-%m-%d %H:%M:%S")
        datetime_format = self.plot_config.get("plot_options", {}).get("datetime_format", "")
        if datetime_format:
            self.datetime_format_edit.setText(datetime_format)
        self.datetime_format_edit.setToolTip(
            "Python datetime format string (e.g., %Y-%m-%d, %Y-%m-%d %H:%M:%S, %m/%d/%Y)\n"
            "Common codes: %Y=year, %m=month, %d=day, %H=hour, %M=minute, %S=second"
        )
        format_layout.addWidget(self.datetime_format_edit)
        format_layout.addStretch()
        datetime_layout.addLayout(format_layout)

        # Date display format
        display_format_layout = QHBoxLayout()
        display_format_layout.addWidget(QLabel("Display Format:"))
        self.datetime_display_format_edit = QLineEdit()
        self.datetime_display_format_edit.setPlaceholderText("%Y-%m-%d")
        datetime_display_format = self.plot_config.get("plot_options", {}).get("datetime_display_format", "")
        if datetime_display_format:
            self.datetime_display_format_edit.setText(datetime_display_format)
        self.datetime_display_format_edit.setToolTip(
            "Format for axis labels (e.g., %Y-%m-%d, %b %d, %m/%d)\n"
            "Leave empty to use matplotlib's automatic formatting"
        )
        display_format_layout.addWidget(self.datetime_display_format_edit)
        display_format_layout.addStretch()
        datetime_layout.addLayout(display_format_layout)

        general_layout.addLayout(datetime_layout)

        general_layout.addStretch()
        tabs.addTab(general_tab, "General")

        # Tab 2: Figure Size & Export
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)

        # Figure size options
        figsize_label = QLabel("<b>Figure Size:</b>")
        export_layout.addWidget(figsize_label)

        figsize_layout = QHBoxLayout()
        figsize_layout.setContentsMargins(0, 5, 0, 10)

        figsize_layout.addWidget(QLabel("Width:"))
        self.figwidth_spin = QDoubleSpinBox()
        self.figwidth_spin.setRange(1.0, 50.0)
        self.figwidth_spin.setSingleStep(0.5)
        self.figwidth_spin.setValue(self.plot_config.get("plot_options", {}).get("figwidth", 8.0))
        self.figwidth_spin.setSuffix(" in")
        self.figwidth_spin.setToolTip("Figure width in inches")
        self.figwidth_spin.setMinimumWidth(100)
        figsize_layout.addWidget(self.figwidth_spin)

        figsize_layout.addWidget(QLabel("Height:"))
        self.figheight_spin = QDoubleSpinBox()
        self.figheight_spin.setRange(1.0, 50.0)
        self.figheight_spin.setSingleStep(0.5)
        self.figheight_spin.setValue(self.plot_config.get("plot_options", {}).get("figheight", 6.0))
        self.figheight_spin.setSuffix(" in")
        self.figheight_spin.setToolTip("Figure height in inches")
        self.figheight_spin.setMinimumWidth(100)
        figsize_layout.addWidget(self.figheight_spin)

        figsize_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 600)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setValue(self.plot_config.get("plot_options", {}).get("dpi", 100))
        self.dpi_spin.setToolTip("Dots per inch for export")
        self.dpi_spin.setMinimumWidth(100)
        figsize_layout.addWidget(self.dpi_spin)

        figsize_layout.addStretch()
        export_layout.addLayout(figsize_layout)

        # Export format options
        export_format_label = QLabel("<b>Export Format:</b>")
        export_layout.addWidget(export_format_label)

        export_format_layout = QHBoxLayout()
        export_format_layout.setContentsMargins(0, 5, 0, 10)
        export_format_layout.addWidget(QLabel("File Format:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["png", "pdf", "svg", "jpg", "eps"])
        current_format = self.plot_config.get("plot_options", {}).get("export_format", "png")
        format_idx = self.export_format_combo.findText(current_format)
        if format_idx >= 0:
            self.export_format_combo.setCurrentIndex(format_idx)
        self.export_format_combo.setToolTip("File format for drag-and-drop export")
        self.export_format_combo.setMinimumWidth(100)
        export_format_layout.addWidget(self.export_format_combo)
        export_format_layout.addStretch()
        export_layout.addLayout(export_format_layout)

        export_layout.addStretch()
        tabs.addTab(export_tab, "Figure & Export")

        # Tab 3: Fonts
        fonts_tab = QWidget()
        fonts_layout = QVBoxLayout(fonts_tab)

        # Font sizes
        font_size_label = QLabel("<b>Font Sizes:</b>")
        fonts_layout.addWidget(font_size_label)

        font_size_layout = QHBoxLayout()
        font_size_layout.setContentsMargins(0, 5, 0, 10)

        font_size_layout.addWidget(QLabel("Title:"))
        self.title_fontsize_spin = QSpinBox()
        self.title_fontsize_spin.setRange(6, 72)
        self.title_fontsize_spin.setValue(self.plot_config.get("plot_options", {}).get("title_fontsize", 12))
        self.title_fontsize_spin.setSuffix(" pt")
        self.title_fontsize_spin.setToolTip("Font size for plot title")
        self.title_fontsize_spin.setMinimumWidth(80)
        font_size_layout.addWidget(self.title_fontsize_spin)

        font_size_layout.addWidget(QLabel("Axis Labels:"))
        self.axis_label_fontsize_spin = QSpinBox()
        self.axis_label_fontsize_spin.setRange(6, 72)
        self.axis_label_fontsize_spin.setValue(self.plot_config.get("plot_options", {}).get("axis_label_fontsize", 10))
        self.axis_label_fontsize_spin.setSuffix(" pt")
        self.axis_label_fontsize_spin.setToolTip("Font size for axis labels")
        self.axis_label_fontsize_spin.setMinimumWidth(80)
        font_size_layout.addWidget(self.axis_label_fontsize_spin)

        font_size_layout.addWidget(QLabel("Tick Labels:"))
        self.tick_fontsize_spin = QSpinBox()
        self.tick_fontsize_spin.setRange(6, 72)
        self.tick_fontsize_spin.setValue(self.plot_config.get("plot_options", {}).get("tick_fontsize", 9))
        self.tick_fontsize_spin.setSuffix(" pt")
        self.tick_fontsize_spin.setToolTip("Font size for axis tick labels")
        self.tick_fontsize_spin.setMinimumWidth(80)
        font_size_layout.addWidget(self.tick_fontsize_spin)

        font_size_layout.addWidget(QLabel("Legend:"))
        self.legend_fontsize_spin = QSpinBox()
        self.legend_fontsize_spin.setRange(6, 72)
        self.legend_fontsize_spin.setValue(self.plot_config.get("plot_options", {}).get("legend_fontsize", 9))
        self.legend_fontsize_spin.setSuffix(" pt")
        self.legend_fontsize_spin.setToolTip("Font size for legend text")
        self.legend_fontsize_spin.setMinimumWidth(80)
        font_size_layout.addWidget(self.legend_fontsize_spin)

        font_size_layout.addStretch()
        fonts_layout.addLayout(font_size_layout)

        # Font family
        font_family_label = QLabel("<b>Font Family:</b>")
        fonts_layout.addWidget(font_family_label)

        font_family_layout = QHBoxLayout()
        font_family_layout.setContentsMargins(0, 5, 0, 10)
        font_family_layout.addWidget(QLabel("Family:"))
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(["serif", "sans-serif", "monospace", "cursive", "fantasy"])
        current_family = self.plot_config.get("plot_options", {}).get("font_family", "serif")
        family_idx = self.font_family_combo.findText(current_family)
        if family_idx >= 0:
            self.font_family_combo.setCurrentIndex(family_idx)
        self.font_family_combo.setToolTip("Font family for all plot text")
        self.font_family_combo.setMinimumWidth(150)
        font_family_layout.addWidget(self.font_family_combo)
        font_family_layout.addStretch()
        fonts_layout.addLayout(font_family_layout)

        fonts_layout.addStretch()
        tabs.addTab(fonts_tab, "Fonts")

        # Tab 4: Series styles
        series_tab = QWidget()
        series_layout = QVBoxLayout(series_tab)

        series_label = QLabel("Configure line style for each data series:")
        series_label.setStyleSheet("font-weight: bold;")
        series_layout.addWidget(series_label)

        # Scroll area for series
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.StyledPanel)

        series_container = QWidget()
        self.series_layout = QVBoxLayout(series_container)
        self.series_layout.setContentsMargins(5, 5, 5, 5)

        # Get Y column indices and names
        y_idxs = self.plot_config.get("y_col_idxs", [])
        series_options = self.plot_config.get("plot_options", {}).get("series", {})

        self.series_widgets = []
        for idx, y_idx in enumerate(y_idxs):
            if y_idx < len(column_names):
                y_name = column_names[y_idx]
                series_widget = self._create_series_widget(
                    y_name, idx, series_options.get(y_name, {})
                )
                self.series_layout.addWidget(series_widget)
                self.series_widgets.append((y_name, series_widget))

        self.series_layout.addStretch()
        scroll.setWidget(series_container)
        series_layout.addWidget(scroll)

        tabs.addTab(series_tab, "Series Styles")

        # Tab 3: Reference Lines
        reflines_tab = QWidget()
        reflines_layout = QVBoxLayout(reflines_tab)

        reflines_label = QLabel("Add horizontal and vertical reference lines:")
        reflines_label.setStyleSheet("font-weight: bold;")
        reflines_layout.addWidget(reflines_label)

        # Scroll area for reference lines
        reflines_scroll = QScrollArea()
        reflines_scroll.setWidgetResizable(True)
        reflines_scroll.setFrameShape(QFrame.StyledPanel)

        reflines_container = QWidget()
        self.reflines_layout = QVBoxLayout(reflines_container)
        self.reflines_layout.setContentsMargins(5, 5, 5, 5)

        # Load existing reference lines
        self.refline_widgets = []
        existing_reflines = self.plot_config.get("plot_options", {}).get("reference_lines", [])
        for refline in existing_reflines:
            self._add_refline_widget(
                refline.get("type", "horizontal"),
                refline.get("value", ""),
                refline.get("color", "#FF0000"),
                refline.get("linestyle", "--"),
                refline.get("linewidth", 1.0),
                refline.get("label", "")
            )

        self.reflines_layout.addStretch()
        reflines_scroll.setWidget(reflines_container)
        reflines_layout.addWidget(reflines_scroll)

        # Buttons to add reference lines
        reflines_buttons = QHBoxLayout()
        add_hline_btn = QPushButton("+ Add Horizontal Line")
        add_hline_btn.clicked.connect(lambda: self._add_refline_widget("horizontal"))
        reflines_buttons.addWidget(add_hline_btn)

        add_vline_btn = QPushButton("+ Add Vertical Line")
        add_vline_btn.clicked.connect(lambda: self._add_refline_widget("vertical"))
        reflines_buttons.addWidget(add_vline_btn)

        reflines_buttons.addStretch()
        reflines_layout.addLayout(reflines_buttons)

        tabs.addTab(reflines_tab, "Reference Lines")

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_series_widget(self, series_name, series_idx, series_options):
        """Create a widget for configuring one series."""
        widget = QFrame()
        widget.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Series name label
        name_label = QLabel(f"<b>{series_name}</b>")
        layout.addWidget(name_label)

        # Color, line style, and marker in one row
        style_layout = QHBoxLayout()

        # Color picker button
        style_layout.addWidget(QLabel("Color:"))
        color_button = QPushButton()
        color_button.setMaximumWidth(80)
        color_button.setMinimumHeight(25)

        # Get current color (hex string or default from cycle)
        current_color = series_options.get("color", self.COLORS[series_idx % len(self.COLORS)])
        # Parse color to QColor
        qcolor = QColor(current_color)
        if not qcolor.isValid():
            # Fallback to first default color if invalid
            qcolor = QColor(self.COLORS[0])

        # Set button style with current color
        color_button.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #999;")
        color_button._color = qcolor

        # Connect to color picker dialog
        def pick_color():
            color = QColorDialog.getColor(color_button._color, self, "Select Color")
            if color.isValid():
                color_button._color = color
                color_button.setStyleSheet(
                    f"background-color: {color.name()}; border: 1px solid #999;"
                )

        color_button.clicked.connect(pick_color)
        style_layout.addWidget(color_button)

        # Line style
        style_layout.addWidget(QLabel("Line Style:"))
        linestyle_combo = QComboBox()
        for i, name in enumerate(self.LINE_STYLE_NAMES):
            linestyle_combo.addItem(name, self.LINE_STYLES[i])
        current_linestyle = series_options.get("linestyle", "-")
        idx = (
            self.LINE_STYLES.index(current_linestyle)
            if current_linestyle in self.LINE_STYLES
            else 0
        )
        linestyle_combo.setCurrentIndex(idx)
        linestyle_combo.setMinimumWidth(100)
        style_layout.addWidget(linestyle_combo)

        # Marker
        style_layout.addWidget(QLabel("Marker:"))
        marker_combo = QComboBox()
        for i, name in enumerate(self.MARKER_NAMES):
            marker_combo.addItem(name, self.MARKERS[i])
        current_marker = series_options.get("marker", "")
        marker_idx = self.MARKERS.index(current_marker) if current_marker in self.MARKERS else 0
        marker_combo.setCurrentIndex(marker_idx)
        marker_combo.setMinimumWidth(100)
        style_layout.addWidget(marker_combo)

        style_layout.addStretch()
        layout.addLayout(style_layout)

        # Line width and Marker size in one row
        size_layout = QHBoxLayout()

        size_layout.addWidget(QLabel("Line Width:"))
        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.5, 5.0)
        width_spin.setSingleStep(0.5)
        width_spin.setValue(series_options.get("linewidth", 1.5))
        width_spin.setMinimumWidth(80)
        size_layout.addWidget(width_spin)

        size_layout.addWidget(QLabel("Marker Size:"))
        markersize_spin = QDoubleSpinBox()
        markersize_spin.setRange(1.0, 20.0)
        markersize_spin.setSingleStep(1.0)
        markersize_spin.setValue(series_options.get("markersize", 6.0))
        markersize_spin.setMinimumWidth(80)
        size_layout.addWidget(markersize_spin)

        size_layout.addStretch()
        layout.addLayout(size_layout)

        # Legend label in its own row
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Legend Label:"))
        label_edit = QLineEdit()
        label_edit.setText(series_options.get("label", series_name))
        label_edit.setPlaceholderText(f"Default: {series_name}")
        label_layout.addWidget(label_edit)
        layout.addLayout(label_layout)

        # Smoothing options
        smooth_layout = QVBoxLayout()
        smooth_checkbox = QCheckBox("Apply Moving Average Smoothing")
        smooth_checkbox.setChecked(series_options.get("smooth", False))
        smooth_layout.addWidget(smooth_checkbox)

        smooth_params_layout = QHBoxLayout()
        smooth_params_layout.addWidget(QLabel("Window Size:"))
        smooth_window_spin = QSpinBox()
        smooth_window_spin.setRange(2, 1000)
        smooth_window_spin.setValue(series_options.get("smooth_window", 5))
        smooth_window_spin.setToolTip("Number of points to average (must be odd for centered averaging)")
        smooth_window_spin.setMinimumWidth(80)
        smooth_params_layout.addWidget(smooth_window_spin)

        smooth_params_layout.addWidget(QLabel("Show:"))
        smooth_mode_combo = QComboBox()
        smooth_mode_combo.addItem("Smoothed Only", "smoothed")
        smooth_mode_combo.addItem("Original + Smoothed", "both")
        smooth_mode_combo.addItem("Original Only (Smoothing Off)", "original")
        current_mode = series_options.get("smooth_mode", "smoothed")
        mode_idx = 0 if current_mode == "smoothed" else (1 if current_mode == "both" else 2)
        smooth_mode_combo.setCurrentIndex(mode_idx)
        smooth_mode_combo.setMinimumWidth(150)
        smooth_params_layout.addWidget(smooth_mode_combo)

        smooth_params_layout.addStretch()
        smooth_layout.addLayout(smooth_params_layout)
        layout.addLayout(smooth_layout)

        # Trend line options
        trend_layout = QVBoxLayout()
        trend_checkbox = QCheckBox("Add Trend Line")
        trend_checkbox.setChecked(series_options.get("trendline", False))
        trend_layout.addWidget(trend_checkbox)

        trend_params_layout = QHBoxLayout()
        trend_params_layout.addWidget(QLabel("Type:"))
        trend_type_combo = QComboBox()
        trend_type_combo.addItem("Linear", "linear")
        trend_type_combo.addItem("Polynomial (degree 2)", "poly2")
        trend_type_combo.addItem("Polynomial (degree 3)", "poly3")
        trend_type_combo.addItem("Polynomial (degree 4)", "poly4")
        current_trend_type = series_options.get("trendline_type", "linear")
        trend_type_idx = 0 if current_trend_type == "linear" else (1 if current_trend_type == "poly2" else (2 if current_trend_type == "poly3" else 3))
        trend_type_combo.setCurrentIndex(trend_type_idx)
        trend_type_combo.setMinimumWidth(150)
        trend_params_layout.addWidget(trend_type_combo)

        trend_params_layout.addWidget(QLabel("Show:"))
        trend_mode_combo = QComboBox()
        trend_mode_combo.addItem("Trend Only", "trend")
        trend_mode_combo.addItem("Original + Trend", "both")
        current_trend_mode = series_options.get("trendline_mode", "both")
        trend_mode_idx = 0 if current_trend_mode == "trend" else 1
        trend_mode_combo.setCurrentIndex(trend_mode_idx)
        trend_mode_combo.setMinimumWidth(150)
        trend_params_layout.addWidget(trend_mode_combo)

        trend_params_layout.addStretch()
        trend_layout.addLayout(trend_params_layout)
        layout.addLayout(trend_layout)

        # Store references
        widget._color_button = color_button
        widget._linestyle_combo = linestyle_combo
        widget._marker_combo = marker_combo
        widget._width_spin = width_spin
        widget._markersize_spin = markersize_spin
        widget._label_edit = label_edit
        widget._smooth_checkbox = smooth_checkbox
        widget._smooth_window_spin = smooth_window_spin
        widget._smooth_mode_combo = smooth_mode_combo
        widget._trend_checkbox = trend_checkbox
        widget._trend_type_combo = trend_type_combo
        widget._trend_mode_combo = trend_mode_combo

        return widget

    def _add_refline_widget(self, line_type, value=None, color=None, linestyle=None, linewidth=None, label=None):
        """Add a reference line configuration widget.

        Args:
            line_type: "horizontal" or "vertical"
            value: Position value (y for horizontal, x for vertical)
            color: Line color (hex string or None for default)
            linestyle: Line style string (default: "solid")
            linewidth: Line width float (default: 1.5)
            label: Optional label for the line
        """
        widget = QFrame()
        widget.setFrameStyle(QFrame.Panel | QFrame.Raised)
        layout = QVBoxLayout(widget)

        # Header row: Type and Remove button
        header_layout = QHBoxLayout()
        type_label = QLabel(f"{'Horizontal' if line_type == 'horizontal' else 'Vertical'} Line")
        type_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(type_label)
        header_layout.addStretch()

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_refline_widget(widget))
        header_layout.addWidget(remove_btn)
        layout.addLayout(header_layout)

        # Value and Color in one row
        value_color_layout = QHBoxLayout()

        value_color_layout.addWidget(QLabel("Value:"))
        value_edit = QLineEdit()
        value_edit.setPlaceholderText("0.0")
        if value is not None:
            value_edit.setText(str(value))
        # Add numeric validator
        value_edit.setValidator(QDoubleValidator())
        value_edit.setMinimumWidth(100)
        value_color_layout.addWidget(value_edit)

        value_color_layout.addWidget(QLabel("Color:"))
        color_button = QPushButton()
        color_button.setMinimumWidth(60)
        color_button.setMaximumWidth(60)
        # Set initial color
        if color:
            initial_color = QColor(color)
        else:
            initial_color = QColor("#000000")  # Black default
        color_button._color = initial_color
        color_button.setStyleSheet(f"background-color: {initial_color.name()};")

        def choose_color():
            color = QColorDialog.getColor(color_button._color, self, "Choose Reference Line Color")
            if color.isValid():
                color_button._color = color
                color_button.setStyleSheet(f"background-color: {color.name()};")

        color_button.clicked.connect(choose_color)
        value_color_layout.addWidget(color_button)

        value_color_layout.addStretch()
        layout.addLayout(value_color_layout)

        # Style, Width, and Label in one row
        style_layout = QHBoxLayout()

        style_layout.addWidget(QLabel("Style:"))
        linestyle_combo = QComboBox()
        linestyle_combo.addItem("Solid", "solid")
        linestyle_combo.addItem("Dashed", "dashed")
        linestyle_combo.addItem("Dash-dot", "dashdot")
        linestyle_combo.addItem("Dotted", "dotted")
        # Set current style
        styles = ["solid", "dashed", "dashdot", "dotted"]
        current_style = linestyle if linestyle in styles else "solid"
        linestyle_combo.setCurrentIndex(styles.index(current_style))
        linestyle_combo.setMinimumWidth(100)
        style_layout.addWidget(linestyle_combo)

        style_layout.addWidget(QLabel("Width:"))
        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.5, 5.0)
        width_spin.setSingleStep(0.5)
        width_spin.setValue(linewidth if linewidth is not None else 1.5)
        width_spin.setMinimumWidth(80)
        style_layout.addWidget(width_spin)

        style_layout.addWidget(QLabel("Label:"))
        label_edit = QLineEdit()
        label_edit.setPlaceholderText("Optional")
        if label:
            label_edit.setText(label)
        label_edit.setMinimumWidth(120)
        style_layout.addWidget(label_edit)

        style_layout.addStretch()
        layout.addLayout(style_layout)

        # Store references and metadata
        widget._line_type = line_type
        widget._value_edit = value_edit
        widget._color_button = color_button
        widget._linestyle_combo = linestyle_combo
        widget._width_spin = width_spin
        widget._label_edit = label_edit

        # Add to layout and tracking list
        self.refline_widgets.append(widget)
        # Insert before the stretch
        self.reflines_layout.insertWidget(self.reflines_layout.count() - 1, widget)

        return widget

    def _remove_refline_widget(self, widget):
        """Remove a reference line widget."""
        if widget in self.refline_widgets:
            self.refline_widgets.remove(widget)
            self.reflines_layout.removeWidget(widget)
            widget.deleteLater()

    def get_plot_config(self):
        """Return updated plot configuration."""
        # Update name
        self.plot_config["name"] = self.name_edit.text()

        # Create or update plot_options
        if "plot_options" not in self.plot_config:
            self.plot_config["plot_options"] = {}

        plot_opts = self.plot_config["plot_options"]
        plot_opts["title"] = self.title_edit.text()
        plot_opts["xlabel"] = self.xlabel_edit.text()
        plot_opts["ylabel"] = self.ylabel_edit.text()
        plot_opts["grid"] = self.grid_checkbox.isChecked()
        plot_opts["legend"] = self.legend_checkbox.isChecked()

        # Save axis limits (convert to float or None)
        def parse_limit(text):
            text = text.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        plot_opts["xlim_min"] = parse_limit(self.xlim_min_edit.text())
        plot_opts["xlim_max"] = parse_limit(self.xlim_max_edit.text())
        plot_opts["ylim_min"] = parse_limit(self.ylim_min_edit.text())
        plot_opts["ylim_max"] = parse_limit(self.ylim_max_edit.text())

        # Save log scale options
        plot_opts["xlog"] = self.xlog_checkbox.isChecked()
        plot_opts["ylog"] = self.ylog_checkbox.isChecked()

        # Save datetime x-axis options
        plot_opts["xaxis_datetime"] = self.xaxis_datetime_checkbox.isChecked()
        plot_opts["datetime_format"] = self.datetime_format_edit.text().strip()
        plot_opts["datetime_display_format"] = self.datetime_display_format_edit.text().strip()

        # Save figure size and export options
        plot_opts["figwidth"] = self.figwidth_spin.value()
        plot_opts["figheight"] = self.figheight_spin.value()
        plot_opts["dpi"] = self.dpi_spin.value()
        plot_opts["export_format"] = self.export_format_combo.currentText()

        # Save font size options
        plot_opts["title_fontsize"] = self.title_fontsize_spin.value()
        plot_opts["axis_label_fontsize"] = self.axis_label_fontsize_spin.value()
        plot_opts["tick_fontsize"] = self.tick_fontsize_spin.value()
        plot_opts["legend_fontsize"] = self.legend_fontsize_spin.value()

        # Save font family option
        plot_opts["font_family"] = self.font_family_combo.currentText()

        # Save reference lines
        ref_lines = []
        for widget in self.refline_widgets:
            value_text = widget._value_edit.text().strip()
            if value_text:  # Only save if value is provided
                try:
                    value = float(value_text)
                    label_text = widget._label_edit.text().strip()
                    ref_lines.append({
                        "type": widget._line_type,
                        "value": value,
                        "color": widget._color_button._color.name(),
                        "linestyle": widget._linestyle_combo.currentData(),
                        "linewidth": widget._width_spin.value(),
                        "label": label_text if label_text else None
                    })
                except ValueError:
                    # Skip invalid values
                    pass
        plot_opts["reference_lines"] = ref_lines

        # Update series options
        series_opts = {}
        for series_name, widget in self.series_widgets:
            series_opts[series_name] = {
                "label": widget._label_edit.text(),
                "color": widget._color_button._color.name(),  # Get hex color from QColor
                "linestyle": widget._linestyle_combo.currentData(),
                "marker": widget._marker_combo.currentData(),
                "linewidth": widget._width_spin.value(),
                "markersize": widget._markersize_spin.value(),
                "smooth": widget._smooth_checkbox.isChecked(),
                "smooth_window": widget._smooth_window_spin.value(),
                "smooth_mode": widget._smooth_mode_combo.currentData(),
                "trendline": widget._trend_checkbox.isChecked(),
                "trendline_type": widget._trend_type_combo.currentData(),
                "trendline_mode": widget._trend_mode_combo.currentData(),
            }
        plot_opts["series"] = series_opts

        return self.plot_config


class CustomSplitter(QSplitter):
    """QSplitter with explicit cursor management for macOS compatibility."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setupCursor()

    def _setupCursor(self):
        """Ensure the splitter handle always has the correct cursor."""
        # Set the cursor based on orientation
        if self.orientation() == Qt.Horizontal:
            cursor = Qt.SplitHCursor
        else:
            cursor = Qt.SplitVCursor

        # Apply cursor to all handles
        for i in range(self.count()):
            handle = self.handle(i)
            if handle:
                handle.setCursor(cursor)

    def addWidget(self, widget):
        """Override to set cursor on handle after widget is added."""
        super().addWidget(widget)
        self._setupCursor()

    def insertWidget(self, index, widget):
        """Override to set cursor on handle after widget is inserted."""
        super().insertWidget(index, widget)
        self._setupCursor()


class HDF5Viewer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VibeHDF5")
        self.resize(900, 600)
        self._original_pixmap = None
        self._current_highlighter = None  # Track current syntax highlighter

        # Central widget: splitter with tree (left) and preview (right)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        splitter = CustomSplitter(self)
        splitter.setHandleWidth(4)  # Make handle slightly wider for easier grabbing
        splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely
        central_layout.addWidget(splitter)

        # Tree view + model (left)
        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Search bar for filtering tree items
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(4, 4, 4, 4)
        search_label = QLabel("Filter:")
        search_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter glob pattern (e.g., *.csv, data*, *test*)")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._on_search_text_changed)
        search_layout.addWidget(self.search_input)

        left_layout.addLayout(search_layout)

        self.tree = DropTreeView(left, viewer=self)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionBehavior(QTreeView.SelectRows)
        self.tree.setHeaderHidden(False)
        left_layout.addWidget(self.tree)

        # Saved plots list widget (below tree)
        saved_plots_label = QLabel("Saved Plots:")
        saved_plots_label.setStyleSheet("font-weight: bold; padding: 4px;")
        left_layout.addWidget(saved_plots_label)

        self.saved_plots_list = DraggablePlotListWidget(self)
        self.saved_plots_list.setMaximumHeight(150)
        self.saved_plots_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.saved_plots_list.customContextMenuRequested.connect(self._on_saved_plots_context_menu)
        self.saved_plots_list.setToolTip("Drag and drop to filesystem to export plot")
        left_layout.addWidget(self.saved_plots_list)

        # Buttons for plot management
        plot_buttons_layout = QHBoxLayout()
        self.btn_save_plot = QPushButton("Save Plot")
        self.btn_save_plot.clicked.connect(self._save_plot_config_dialog)
        self.btn_save_plot.setEnabled(False)
        plot_buttons_layout.addWidget(self.btn_save_plot)

        self.btn_edit_plot_options = QPushButton("Edit Options")
        self.btn_edit_plot_options.clicked.connect(self._edit_plot_options_dialog)
        self.btn_edit_plot_options.setEnabled(False)
        plot_buttons_layout.addWidget(self.btn_edit_plot_options)

        self.btn_delete_plot = QPushButton("Delete")
        self.btn_delete_plot.clicked.connect(self._delete_plot_config)
        self.btn_delete_plot.setEnabled(False)
        plot_buttons_layout.addWidget(self.btn_delete_plot)

        left_layout.addLayout(plot_buttons_layout)

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
        right_splitter = CustomSplitter(Qt.Vertical, right)
        right_splitter.setHandleWidth(4)  # Make handle slightly wider for easier grabbing
        right_splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely
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

        self.btn_show_statistics = QPushButton("Statistics...")
        self.btn_show_statistics.clicked.connect(self._show_statistics_dialog)
        filter_panel_layout.addWidget(self.btn_show_statistics)

        self.btn_configure_sort = QPushButton("Sort...")
        self.btn_configure_sort.clicked.connect(self._configure_sort_dialog)
        filter_panel_layout.addWidget(self.btn_configure_sort)

        self.btn_clear_sort = QPushButton("Clear Sort")
        self.btn_clear_sort.clicked.connect(self._clear_sort)
        self.btn_clear_sort.setEnabled(False)
        filter_panel_layout.addWidget(self.btn_clear_sort)

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
        # Connect scrollbar for lazy loading
        self.preview_table.verticalScrollBar().valueChanged.connect(self._on_table_scroll)
        content_layout.addWidget(self.preview_table)

        # Lazy loading state variables
        self._table_batch_size = 1000  # Load rows in batches
        self._table_loaded_rows = 0  # Track how many rows are loaded
        self._table_is_loading = False  # Prevent concurrent loads

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

        self.plot_figure = Figure(figsize=(8, 6))
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, plot_widget)

        # Add toolbar first, then canvas
        plot_layout.addWidget(self.plot_toolbar)
        plot_layout.addWidget(self.plot_canvas)

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

        # Track saved plot configurations for current CSV group
        self._saved_plots: list[dict] = []  # List of plot config dictionaries

        # Connect saved plots list selection changed
        self.saved_plots_list.itemSelectionChanged.connect(self._on_saved_plot_selection_changed)

        # Track current search pattern
        self._search_pattern: str = ""

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
                            is_csv = (
                                "source_type" in obj.attrs and obj.attrs["source_type"] == "csv"
                            )
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
                            is_csv = (
                                "source_type" in obj.attrs and obj.attrs["source_type"] == "csv"
                            )
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

        folder_name, ok = QInputDialog.getText(
            self, "New Folder", f"Enter folder name to create in {target_group}:"
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
                        self, "Already Exists", f"Group '{new_group_path}' already exists."
                    )
                    return
                h5.create_group(new_group_path)

            # Reload the tree and expand to show the new folder
            self.model.load_file(fpath)
            self.tree.expandToDepth(2)
            self.statusBar().showMessage(f"Created folder: {new_group_path}", 5000)

        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to create folder: {exc}")

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
                if (
                    source_parent
                    and source_parent != "/"
                    and isinstance(h5[source_parent], h5py.Group)
                ):
                    parent_grp = h5[source_parent]
                    if (
                        "source_type" in parent_grp.attrs
                        and parent_grp.attrs["source_type"] == "csv"
                    ):
                        QMessageBox.warning(
                            self, "Invalid Move", "Cannot move items out of CSV groups."
                        )
                        return False

                # Prevent moving into CSV groups
                if (
                    target_group
                    and target_group != "/"
                    and isinstance(h5[target_group], h5py.Group)
                ):
                    grp = h5[target_group]
                    if "source_type" in grp.attrs and grp.attrs["source_type"] == "csv":
                        QMessageBox.warning(
                            self, "Invalid Target", "Cannot move items into CSV groups."
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
            QMessageBox.critical(self, "Move Failed", f"Failed to move item: {exc}")
            return False

    def _add_items_batch(
        self, files: list[str], folders: list[str], target_group: str
    ) -> tuple[int, list[str]]:
        fpath = self.model.filepath
        if not fpath:
            return 0, ["No HDF5 file loaded"]
        errors: list[str] = []
        added = 0
        try:
            with h5py.File(fpath, "r+") as h5:
                # Final safety: never allow writing into a CSV-derived group
                try:
                    if (
                        target_group
                        and target_group != "/"
                        and isinstance(h5[target_group], h5py.Group)
                    ):
                        grp = h5[target_group]
                        if "source_type" in grp.attrs and grp.attrs["source_type"] == "csv":
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
                    h5_path = (
                        posixpath.join(target_group, name) if target_group != "/" else "/" + name
                    )
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
        if disk_path.lower().endswith(".csv"):
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
            compressed = gzip.compress(data.encode("utf-8"), compresslevel=9)
            ds = f.create_dataset(h5_path, data=np.frombuffer(compressed, dtype="uint8"))
            # Mark as compressed text so we can decompress on read
            ds.attrs["compressed"] = "gzip"
            ds.attrs["original_encoding"] = "utf-8"
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
        ds.attrs["compressed"] = "gzip"
        ds.attrs["original_encoding"] = "binary"

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
        if group_path.lower().endswith(".csv"):
            group_path = group_path[:-4]

        # Ensure parent groups exist
        parent = os.path.dirname(group_path).replace("\\", "/")
        if parent and parent != "/":
            f.require_group(parent)

        # Create a group for the CSV data
        grp = f.create_group(group_path)

        # Add metadata about the source file
        grp.attrs["source_file"] = os.path.basename(disk_path)
        grp.attrs["source_type"] = "csv"
        grp.attrs["column_names"] = list(df.columns)

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
            ds_name = base if base else "unnamed_column"
            # Ensure uniqueness within the group
            if ds_name in used_names:
                i = 2
                while f"{ds_name}_{i}" in used_names or f"{ds_name}_{i}" in grp:
                    i += 1
                ds_name = f"{ds_name}_{i}"
            used_names.add(ds_name)
            column_dataset_names.append(ds_name)

            # Convert pandas Series to numpy array with appropriate dtype
            if col_data.dtype == "object":
                # For object dtype, convert to Python list then create dataset
                # This avoids numpy unicode string issues
                try:
                    # Convert to Python strings
                    str_list = [str(x) for x in col_data.values]
                    # Use gzip compression for string columns
                    grp.create_dataset(
                        ds_name,
                        data=str_list,
                        dtype=h5py.string_dtype(encoding="utf-8"),
                        compression="gzip",
                        compression_opts=4  # Compression level 1-9 (4 is good balance)
                    )
                except Exception:  # noqa: BLE001
                    # Fallback: convert to bytes
                    str_list = [str(x) for x in col_data.values]
                    grp.create_dataset(
                        ds_name, data=str_list, dtype=h5py.string_dtype(encoding="utf-8")
                    )
            else:
                # Numeric or other numpy-supported dtypes with compression
                # Use chunking to enable compression and improve I/O for partial reads
                chunk_size = min(10000, len(col_data))  # Reasonable chunk size
                grp.create_dataset(
                    ds_name,
                    data=col_data.values,
                    compression="gzip",
                    compression_opts=4,  # Compression level 1-9 (4 is good balance)
                    chunks=(chunk_size,) if len(col_data) > 1000 else None
                )

        # Persist the actual dataset names used for each column (same order as column_names)
        progress.setLabelText("Finalizing CSV import...")
        progress.setValue(95)
        QApplication.processEvents()

        try:
            grp.attrs["column_dataset_names"] = np.array(column_dataset_names, dtype=object)
        except Exception:  # noqa: BLE001
            # Fallback to list assignment if dtype=object attr not permitted
            grp.attrs["column_dataset_names"] = column_dataset_names

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
        if not filepath.endswith((".h5", ".hdf5")):
            filepath += ".h5"

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

    # Search/Filter handling
    def _on_search_text_changed(self, text: str) -> None:
        """Handle search text changes and filter the tree view."""
        self._search_pattern = text.strip()
        self._apply_tree_filter()

    def _apply_tree_filter(self) -> None:
        """Apply the search filter to the tree view."""
        if not self._search_pattern:
            # Show all items if search is empty
            self._set_all_items_visible(self.model.invisibleRootItem(), True)
            return

        # Hide all items first
        self._set_all_items_visible(self.model.invisibleRootItem(), False)

        # Show items matching the pattern and their parents
        self._filter_items_recursive(self.model.invisibleRootItem(), self._search_pattern)

    def _set_all_items_visible(self, parent_item, visible: bool) -> None:
        """Recursively set visibility of all items in the tree."""
        for row in range(parent_item.rowCount()):
            child_item = parent_item.child(row, 0)
            if child_item:
                # Get the index for this item
                index = child_item.index()
                self.tree.setRowHidden(index.row(), index.parent(), not visible)
                # Recursively process children
                self._set_all_items_visible(child_item, visible)

    def _filter_items_recursive(self, parent_item, pattern: str) -> bool:
        """Recursively filter items based on glob pattern.

        Returns True if this item or any of its children match the pattern.
        """
        has_visible_child = False

        for row in range(parent_item.rowCount()):
            child_item = parent_item.child(row, 0)
            if not child_item:
                continue

            # Get the item name
            item_name = child_item.text()

            # Check if any children match (recursive)
            child_has_match = self._filter_items_recursive(child_item, pattern)

            # Check if this item matches the pattern
            # If pattern contains a slash, match against the full path from root
            if '/' in pattern:
                # Build the full path for this item
                full_path = self._get_item_path(child_item)
                # Strip leading slash for matching (e.g., "/folder/file.png" -> "folder/file.png")
                if full_path.startswith('/'):
                    full_path = full_path[1:]
                item_matches = fnmatch.fnmatch(full_path.lower(), pattern.lower())
            else:
                # Match against just the item name
                item_matches = fnmatch.fnmatch(item_name.lower(), pattern.lower())

            # Show item if it matches OR if any of its children match
            should_show = item_matches or child_has_match

            # Get the index for this item
            index = child_item.index()
            self.tree.setRowHidden(index.row(), index.parent(), not should_show)

            if should_show:
                has_visible_child = True

        return has_visible_child

    def _get_item_path(self, item) -> str:
        """Build the full path of an item from root to the item."""
        path_parts = []
        current = item
        while current is not None:
            # Skip the root invisible item
            if current.parent() is None:
                break
            path_parts.append(current.text())
            current = current.parent()

        # Reverse to get path from root to item
        path_parts.reverse()
        # Skip the first element (filename) to get HDF5-like path
        if len(path_parts) > 1:
            return '/' + '/'.join(path_parts[1:])
        return '/'

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
                        if "source_type" in grp.attrs and grp.attrs["source_type"] == "csv":
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
            QMessageBox.warning(
                self, "Refresh failed", f"Deleted, but failed to refresh view: {exc}"
            )

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
        if dspath.lower().endswith(".png"):
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
                    if "compressed" in obj.attrs and obj.attrs["compressed"] == "gzip":
                        encoding = obj.attrs.get("original_encoding", "utf-8")
                        if isinstance(encoding, bytes):
                            encoding = encoding.decode("utf-8")
                        if (
                            encoding == "binary"
                            and isinstance(data, np.ndarray)
                            and data.dtype == np.uint8
                        ):
                            # Decompress the binary data
                            compressed_bytes = data.tobytes()
                            img_bytes = gzip.decompress(compressed_bytes)
                        elif isinstance(data, bytes):
                            img_bytes = data
                        elif hasattr(data, "tobytes"):
                            img_bytes = data.tobytes()
                        else:
                            self._set_preview_text("Dataset is not a valid PNG byte array.")
                            self.preview_edit.setVisible(True)
                            self.preview_image.setVisible(False)
                            self._hide_attributes()
                            return
                    elif isinstance(data, bytes):
                        img_bytes = data
                    elif hasattr(data, "tobytes"):
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
                    self.preview_edit.document(), language=language
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
                if "source_type" in grp.attrs and grp.attrs["source_type"] == "csv":
                    # Track current CSV group for plotting
                    self._current_csv_group_path = grouppath
                    self._show_csv_table(grp)
                    self._update_plot_action_enabled()
                else:
                    self._current_csv_group_path = None
                    self._saved_plots = []
                    self._refresh_saved_plots_list()
                    self._set_preview_text("(No content to display)")
                    # Show attributes for the group
                    self._show_attributes(grp)
        except Exception as exc:
            self._set_preview_text(f"Error reading group:\n{exc}")
            self._hide_attributes()

    def _get_th_location(self, ds_key, grp):
        """We allow an optional 'Time History' group for CSV columns."""
        OPTIONAL_GROUP_FOR_COLUMNS = "Time History"
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
            if "column_names" in grp.attrs:
                try:
                    col_names = [str(c) for c in list(grp.attrs["column_names"])]
                except Exception:  # noqa: BLE001
                    col_names = list(grp.keys())
            else:
                col_names = list(grp.keys())

            # Optional mapping of columns to actual dataset names
            col_ds_names: list[str] | None = None
            if "column_dataset_names" in grp.attrs:
                try:
                    col_ds_names = [str(c) for c in list(grp.attrs["column_dataset_names"])]
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
                            # Decode byte strings to UTF-8 strings for display
                            if data.dtype.kind == "S":
                                # Byte strings - decode to UTF-8
                                data = np.array([
                                    v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                                    for v in data
                                ], dtype=object)
                            elif data.dtype.kind == "O":
                                # Object dtype - could be mixed, handle bytes if present
                                data = np.array([
                                    v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                                    for v in data
                                ], dtype=object)
                            data_dict[col_name] = data
                            max_rows = max(max_rows, len(data))
                        else:
                            # Scalar dataset - handle bytes
                            if isinstance(data, bytes):
                                data = data.decode("utf-8", errors="replace")
                            data_dict[col_name] = [data]
                            max_rows = max(max_rows, 1)

            if not data_dict:
                progress.close()
                self._set_preview_text("(No datasets found in CSV group)")
                return

            # Store full data for filtering and lazy loading
            self._csv_data_dict = data_dict
            self._csv_column_names = col_names
            self._csv_total_rows = max_rows
            self._csv_sort_specs = []  # Initialize sort specs

            # Setup table - disable updates for performance
            progress.setLabelText("Setting up table...")
            progress.setValue(75)
            QApplication.processEvents()

            self.preview_table.setUpdatesEnabled(False)
            self.preview_table.setSortingEnabled(False)
            self.preview_table.clear()
            self.preview_table.setRowCount(max_rows)  # Set full row count for scrollbar
            self.preview_table.setColumnCount(len(col_names))
            self.preview_table.setHorizontalHeaderLabels(col_names)

            # Initially load only first batch of rows for performance
            initial_batch = min(self._table_batch_size, max_rows)
            progress.setLabelText(f"Loading initial {initial_batch} rows...")
            progress.setValue(80)
            QApplication.processEvents()

            self._populate_table_rows(0, initial_batch, data_dict, col_names)
            self._table_loaded_rows = initial_batch

            # Resize columns to content based on initial batch
            progress.setLabelText("Resizing columns...")
            progress.setValue(95)
            QApplication.processEvents()

            self.preview_table.resizeColumnsToContents()

            # Re-enable updates and sorting
            self.preview_table.setUpdatesEnabled(True)
            # self.preview_table.setSortingEnabled(True)  # don't want sorting since it interfers with selecting columns for plotting.

            # Show table, hide others
            progress.setValue(100)
            self.preview_table.setVisible(True)
            self.preview_edit.setVisible(False)
            self.preview_image.setVisible(False)

            # Show filter panel for CSV tables
            self.filter_panel.setVisible(True)

            # Show attributes for the CSV group
            self._show_attributes(grp)

            # Show message about lazy loading
            if max_rows > initial_batch:
                self.statusBar().showMessage(
                    f"Loaded {initial_batch:,} of {max_rows:,} rows (more will load as you scroll)", 8000
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

            # Load saved sort from HDF5 group
            saved_sort = self._load_sort_from_hdf5(grp)
            if saved_sort:
                self._csv_sort_specs = saved_sort
                self.btn_clear_sort.setEnabled(True)
                self.statusBar().showMessage(
                    f"Loaded sort by {len(saved_sort)} column(s) from HDF5 file", 5000
                )

            # Load saved plot configurations from HDF5 group
            self._load_plot_configs_from_hdf5(grp)

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

    def _populate_table_rows(self, start_row: int, end_row: int, data_dict: dict, col_names: list[str]) -> None:
        """Populate table rows from start_row to end_row (exclusive)."""
        for col_idx, col_name in enumerate(col_names):
            if col_name in data_dict:
                col_data = data_dict[col_name]
                # Convert column slice to strings
                if isinstance(col_data, np.ndarray):
                    # Handle different data types
                    if col_data.dtype.kind == "S":
                        # Byte strings - decode to UTF-8
                        str_data = [
                            v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                            for v in col_data[start_row:end_row]
                        ]
                    elif col_data.dtype.kind == "U":
                        # Unicode strings - already strings, just convert
                        str_data = [str(v) for v in col_data[start_row:end_row]]
                    elif col_data.dtype.kind == "O":
                        # Object dtype - could be mixed, handle bytes if present
                        str_data = [
                            v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                            for v in col_data[start_row:end_row]
                        ]
                    else:
                        # Numeric or other types - use numpy's string conversion
                        str_data = np.char.mod("%s", col_data[start_row:end_row])
                else:
                    # Handle bytes in non-array data
                    if isinstance(col_data, bytes):
                        str_data = [col_data.decode("utf-8", errors="replace")]
                    else:
                        str_data = [str(v) if not isinstance(v, bytes) else v.decode("utf-8", errors="replace") for v in col_data[start_row:end_row]]

                # Set items in table
                for i, value_str in enumerate(str_data):
                    row_idx = start_row + i
                    item = QTableWidgetItem(value_str)
                    self.preview_table.setItem(row_idx, col_idx, item)

    def _on_table_scroll(self, value: int) -> None:
        """Handle table scroll events to load more rows as needed."""
        if self._table_is_loading:
            return  # Already loading, skip

        if not hasattr(self, '_csv_total_rows') or not hasattr(self, '_csv_data_dict'):
            return  # No CSV data loaded

        # Check if we need to load more rows
        if self._table_loaded_rows >= self._csv_total_rows:
            return  # All rows already loaded

        # Get visible range
        scrollbar = self.preview_table.verticalScrollBar()
        max_value = scrollbar.maximum()
        if max_value == 0:
            return

        # Calculate approximate visible row based on scroll position
        scroll_ratio = value / max_value
        approx_visible_row = int(scroll_ratio * self._csv_total_rows)

        # Add buffer rows above and below visible area
        buffer_rows = 500
        target_row = min(approx_visible_row + buffer_rows, self._csv_total_rows)

        # If the target row is beyond what we've loaded, load up to that point
        if target_row > self._table_loaded_rows:
            self._load_rows_up_to(target_row)

    def _load_rows_up_to(self, target_row: int) -> None:
        """Load all rows from current position up to target_row."""
        if self._table_is_loading:
            return

        if self._table_loaded_rows >= target_row:
            return

        self._table_is_loading = True

        try:
            # Load all rows from current position to target in one go
            start_row = self._table_loaded_rows
            end_row = min(target_row, self._csv_total_rows)

            # Disable updates during batch load for better performance
            self.preview_table.setUpdatesEnabled(False)

            # Populate all rows up to target
            self._populate_table_rows(start_row, end_row, self._csv_data_dict, self._csv_column_names)

            # Update loaded count
            self._table_loaded_rows = end_row

            # Re-enable updates
            self.preview_table.setUpdatesEnabled(True)

            # Update status bar
            if self._table_loaded_rows < self._csv_total_rows:
                self.statusBar().showMessage(
                    f"Loaded {self._table_loaded_rows:,} of {self._csv_total_rows:,} rows", 2000
                )
            else:
                self.statusBar().showMessage(
                    f"All {self._csv_total_rows:,} rows loaded", 3000
                )
        except Exception as exc:
            self.statusBar().showMessage(f"Error loading rows: {exc}", 5000)
        finally:
            self._table_is_loading = False

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

        # Also update plot management buttons
        self._update_plot_buttons_state()

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
                if "column_names" in grp.attrs:
                    try:
                        orig = [str(c) for c in list(grp.attrs["column_names"])]
                    except Exception:  # noqa: BLE001
                        orig = []
                    ds_names: list[str] | None = None
                    if "column_dataset_names" in grp.attrs:
                        try:
                            ds_names = [str(c) for c in list(grp.attrs["column_dataset_names"])]
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
                        # Decode byte strings to UTF-8 strings
                        if arr.dtype.kind == "S":
                            # Byte strings - decode to UTF-8
                            arr = np.array([
                                v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                                for v in arr
                            ], dtype=object)
                        elif arr.dtype.kind == "O":
                            # Object dtype - could be mixed, handle bytes if present
                            arr = np.array([
                                v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                                for v in arr
                            ], dtype=object)
                    else:
                        # Handle scalar bytes
                        if isinstance(data, bytes):
                            data = data.decode("utf-8", errors="replace")
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
            if self.preview_table.horizontalHeaderItem(i) is not None
            else f"col_{i}"
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
            x_arr = col_data[x_name]
            # Ensure 1-D
            x_arr = x_arr.ravel()
            min_len = min(len(x_arr), *(len(col_data.get(n, [])) for n in y_names if n in col_data))
            if min_len <= 0:
                QMessageBox.warning(self, "Plot", "No data to plot.")
                return

            # Check if x_arr contains strings (automatic date detection)
            x_is_string = False
            xaxis_datetime = False
            if len(x_arr) > 0:
                first_val = x_arr[0]
                x_is_string = isinstance(first_val, str) or (
                    hasattr(first_val, "dtype") and first_val.dtype.kind in ("U", "O")
                )

            # If x-axis is strings, try auto-parsing as dates
            if x_is_string:
                try:
                    # Try to parse as datetime without explicit format (pandas will infer)
                    x_data = pd.to_datetime(pd.Series(x_arr[:min_len]), errors="coerce")
                    # Check if parsing was successful (not all NaT/null)
                    valid_dates = x_data.notna()
                    if valid_dates.sum() > 0:
                        # Successfully parsed as dates - use datetime mode
                        # Convert to matplotlib date numbers using pandas
                        # Convert pandas datetime to matplotlib float dates
                        x_num = np.array([mdates.date2num(d) if pd.notna(d) else np.nan
                                         for d in x_data])
                        xaxis_datetime = True
                    else:
                        # Parsing failed - treat as categorical/text
                        # Use integer indices for x-axis
                        x_num = np.arange(min_len, dtype=float)
                        xaxis_datetime = False
                except Exception:
                    # If auto-parsing fails, use integer indices
                    x_num = np.arange(min_len, dtype=float)
                    xaxis_datetime = False
            else:
                # Try numeric conversion
                x_num = (
                    pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce")
                    .astype(float)
                    .to_numpy()
                )

            # Clear the previous plot
            self.plot_figure.clear()

            # Create a subplot
            ax = self.plot_figure.add_subplot(111)

            # Disable offset notation on axes
            ax.ticklabel_format(useOffset=False)

            any_plotted = False
            for y_name in y_names:
                if y_name not in col_data:
                    continue
                y_arr = col_data[y_name].ravel()[:min_len]
                y_num = pd.to_numeric(pd.Series(y_arr), errors="coerce").astype(float).to_numpy()

                valid = np.isfinite(x_num) & np.isfinite(y_num)
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
                    filtered_rows = (
                        len(self._csv_filtered_indices)
                        if self._csv_filtered_indices is not None
                        else 0
                    )
                    title += f" ({filtered_rows}/{total_rows} rows, filtered)"
                ax.set_title(title)
            except Exception:
                pass

            ax.set_xlabel(x_name)
            ax.set_ylabel(", ".join(y_names))
            ax.legend()
            ax.grid(True)

            # Format datetime x-axis if dates were detected
            if xaxis_datetime:
                # Use automatic date formatting
                ax.xaxis.set_major_locator(AutoDateLocator())
                # Rotate labels for better readability
                self.plot_figure.autofmt_xdate()
            elif x_is_string and not xaxis_datetime:
                # X-axis is categorical strings - set string labels on integer positions
                # Limit labels to avoid overcrowding
                num_points = min(min_len, len(x_arr))
                if num_points <= 50:
                    # Show all labels if not too many
                    ax.set_xticks(np.arange(num_points))
                    ax.set_xticklabels(x_arr[:num_points], rotation=45, ha="right")
                else:
                    # Show subset of labels to avoid overcrowding
                    step = max(1, num_points // 20)  # Show ~20 labels
                    indices = np.arange(0, num_points, step)
                    ax.set_xticks(indices)
                    ax.set_xticklabels([x_arr[i] for i in indices], rotation=45, ha="right")

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

    def _show_statistics_dialog(self):
        """Open dialog to show statistics for CSV columns."""
        if not self._csv_column_names or not self._csv_data_dict:
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        # Use filtered indices if available
        filtered_indices = self._csv_filtered_indices if hasattr(self, '_csv_filtered_indices') else None

        dialog = ColumnStatisticsDialog(
            self._csv_column_names,
            self._csv_data_dict,
            filtered_indices,
            self
        )
        dialog.exec()

    def _clear_filters(self):
        """Clear all active filters and show full dataset."""
        self._csv_filters = []
        self._save_filters_to_hdf5()
        self._apply_filters()

    def _configure_sort_dialog(self):
        """Open dialog to configure column sorting."""
        if not self._csv_column_names:
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        dialog = ColumnSortDialog(self._csv_column_names, self)
        dialog.set_sort_specs(self._csv_sort_specs)

        if dialog.exec() == QDialog.Accepted:
            self._csv_sort_specs = dialog.get_sort_specs()
            self._save_sort_to_hdf5()
            self._apply_sort()

    def _clear_sort(self):
        """Clear all sorting and display data in original order."""
        self._csv_sort_specs = []
        self._save_sort_to_hdf5()
        self._apply_sort()

    def _save_sort_to_hdf5(self):
        """Save current sort specifications to the HDF5 file as a JSON attribute."""
        if not self._current_csv_group_path or not self.model or not self.model.filepath:
            return

        try:
            with h5py.File(self.model.filepath, "r+") as h5:
                if self._current_csv_group_path in h5:
                    grp = h5[self._current_csv_group_path]
                    if isinstance(grp, h5py.Group):
                        if self._csv_sort_specs:
                            # Convert sort specs to JSON string
                            # Format: list of [column_name, ascending]
                            sort_json = json.dumps(self._csv_sort_specs)
                            grp.attrs["csv_sort"] = sort_json
                            self.statusBar().showMessage(
                                f"Saved sort by {len(self._csv_sort_specs)} column(s) to HDF5 file", 3000
                            )
                        else:
                            # Remove sort attribute if no sort specs
                            if "csv_sort" in grp.attrs:
                                del grp.attrs["csv_sort"]
                            self.statusBar().showMessage("Cleared sort from HDF5 file", 3000)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not save sort to HDF5: {exc}")

    def _load_sort_from_hdf5(self, grp: h5py.Group):
        """Load sort specifications from the HDF5 group attributes."""
        try:
            if "csv_sort" in grp.attrs:
                sort_json = grp.attrs["csv_sort"]
                if isinstance(sort_json, bytes):
                    sort_json = sort_json.decode("utf-8")
                sort_specs = json.loads(sort_json)
                # Validate format
                if isinstance(sort_specs, list):
                    # Ensure each spec is a 2-element list
                    valid_specs = []
                    for spec in sort_specs:
                        if isinstance(spec, list) and len(spec) == 2:
                            valid_specs.append((spec[0], spec[1]))
                    return valid_specs
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not load sort from HDF5: {exc}")
        return []

    def _save_filters_to_hdf5(self):
        """Save current filters to the HDF5 file as a JSON attribute."""
        if not self._current_csv_group_path or not self.model or not self.model.filepath:
            return

        try:

            with h5py.File(self.model.filepath, "r+") as h5:
                if self._current_csv_group_path in h5:
                    grp = h5[self._current_csv_group_path]
                    if isinstance(grp, h5py.Group):
                        if self._csv_filters:
                            # Convert filters to JSON string
                            # Format: list of [column_name, operator, value]
                            filters_json = json.dumps(self._csv_filters)
                            grp.attrs["csv_filters"] = filters_json
                            self.statusBar().showMessage(
                                f"Saved {len(self._csv_filters)} filter(s) to HDF5 file", 3000
                            )
                        else:
                            # Remove filter attribute if no filters
                            if "csv_filters" in grp.attrs:
                                del grp.attrs["csv_filters"]
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
            if "csv_filters" in grp.attrs:

                filters_json = grp.attrs["csv_filters"]
                if isinstance(filters_json, bytes):
                    filters_json = filters_json.decode("utf-8")
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

    def _apply_sort(self):
        """Apply current sort specifications to the CSV table."""
        if not self._csv_data_dict or not self._csv_column_names:
            return

        # Update sort button state
        if self._csv_sort_specs:
            self.btn_clear_sort.setEnabled(True)
        else:
            self.btn_clear_sort.setEnabled(False)

        # After changing sort, reapply filters to update display
        self._apply_filters()

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

        # Apply sorting if specified
        if self._csv_sort_specs:
            # Build list of columns and orders for sorting
            sort_columns = []
            sort_orders = []

            for col_name, ascending in self._csv_sort_specs:
                if col_name in self._csv_data_dict:
                    sort_columns.append(col_name)
                    sort_orders.append(ascending)

            if sort_columns:
                # Use pandas for multi-column sorting (handles mixed types better)
                try:
                    # Create a DataFrame from the filtered data
                    sort_data = {}
                    for col_name in sort_columns:
                        col_data = self._csv_data_dict[col_name][filtered_indices]
                        sort_data[col_name] = col_data

                    df = pd.DataFrame(sort_data)

                    # Sort by multiple columns
                    df_sorted = df.sort_values(
                        by=sort_columns,
                        ascending=sort_orders,
                        na_position='last'
                    )

                    # Get the sorted indices and apply to filtered_indices
                    sorted_positions = df_sorted.index.values
                    filtered_indices = filtered_indices[sorted_positions]

                except Exception as e:
                    print(f"Warning: Could not sort data: {e}")
                    # Continue with unsorted data

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
                if filtered_data.dtype.kind == "S" or filtered_data.dtype.kind == "U":
                    str_data = [
                        str(v) if not isinstance(v, bytes) else v.decode("utf-8", errors="replace")
                        for v in filtered_data
                    ]
                else:
                    str_data = np.char.mod("%s", filtered_data)
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
                        col_numeric = pd.to_numeric(pd.Series(col_data), errors="coerce")
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
                    # Not numeric, try date/time comparison for string columns
                    if operator in ["==", "!=", ">", ">=", "<", "<="]:
                        try:
                            # Convert to string array first
                            if isinstance(col_data, np.ndarray):
                                str_array = np.array(
                                    [
                                        str(v) if not isinstance(v, bytes) else v.decode("utf-8", errors="replace")
                                        for v in col_data
                                    ]
                                )
                            else:
                                str_array = np.array([str(v) for v in col_data])

                            # Try to parse as datetime
                            col_dates = pd.to_datetime(pd.Series(str_array), errors="coerce")
                            value_date = pd.to_datetime(value_str, errors="coerce")

                            # Check if parsing was successful (not all NaT and value is not NaT)
                            if not pd.isna(value_date) and not col_dates.isna().all():
                                # Use datetime comparison
                                if operator == "==":
                                    return col_dates == value_date
                                elif operator == "!=":
                                    return col_dates != value_date
                                elif operator == ">":
                                    return col_dates > value_date
                                elif operator == ">=":
                                    return col_dates >= value_date
                                elif operator == "<":
                                    return col_dates < value_date
                                elif operator == "<=":
                                    return col_dates <= value_date
                        except (ValueError, TypeError):
                            # Fall back to string comparison
                            pass

            # String-based operations
            if isinstance(col_data, np.ndarray):
                # Convert to string array for comparison
                str_array = np.array(
                    [
                        str(v) if not isinstance(v, bytes) else v.decode("utf-8", errors="replace")
                        for v in col_data
                    ]
                )
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

    # ========== Plot Configuration Management ==========

    def _save_plot_config_dialog(self):
        """Open dialog to save current plot configuration."""
        if not self._current_csv_group_path or not self.preview_table.isVisible():
            QMessageBox.information(
                self, "No CSV Data", "Load a CSV group and create a plot first."
            )
            return

        # Get currently selected columns
        sel_cols = self._get_selected_column_indices()
        if len(sel_cols) < 2:
            QMessageBox.information(
                self,
                "No Plot Selection",
                "Select at least two columns (X and Y) before saving a plot configuration.",
            )
            return

        # Determine X and Y columns (same logic as plot_selected_columns)
        current_col = self.preview_table.currentColumn()
        x_idx = current_col if current_col in sel_cols else min(sel_cols)
        y_idxs = [c for c in sel_cols if c != x_idx]

        # Prompt for plot name
        plot_name, ok = QInputDialog.getText(
            self,
            "Save Plot Configuration",
            "Enter a name for this plot configuration:",
            QLineEdit.Normal,
            f"Plot {len(self._saved_plots) + 1}",
        )

        if not ok or not plot_name:
            return

        # Store the complete filtered indices array to properly handle non-contiguous filtering
        if self._csv_filtered_indices is not None and len(self._csv_filtered_indices) > 0:
            # Store as a list for JSON serialization
            filtered_indices = self._csv_filtered_indices.tolist()
            start_row = int(self._csv_filtered_indices[0])
            end_row = int(self._csv_filtered_indices[-1])
        else:
            # No filtering - use full range
            max_rows = (
                max(len(self._csv_data_dict[col]) for col in self._csv_data_dict)
                if self._csv_data_dict
                else 0
            )
            filtered_indices = None
            start_row = 0
            end_row = max_rows - 1 if max_rows > 0 else 0

        # Create plot configuration dictionary

        # Get column names
        column_names = [
            self.preview_table.horizontalHeaderItem(i).text()
            if self.preview_table.horizontalHeaderItem(i) is not None
            else f"col_{i}"
            for i in range(self.preview_table.columnCount())
        ]

        plot_config = {
            "name": plot_name,
            "csv_group_path": self._current_csv_group_path,
            "column_names": column_names,
            "x_col_idx": x_idx,
            "y_col_idxs": y_idxs,
            "filtered_indices": filtered_indices,  # Store actual filtered row indices
            "start_row": start_row,  # Keep for backward compatibility
            "end_row": end_row,  # Keep for backward compatibility
            "timestamp": time.time(),
            "plot_options": {
                "title": "",
                "xlabel": "",
                "ylabel": "",
                "grid": True,
                "legend": True,
                "series": {},  # Will be populated with per-series styles in the Edit Options dialog
            },
        }

        # Add to local list
        self._saved_plots.append(plot_config)

        # Save to HDF5
        self._save_plot_configs_to_hdf5()

        # Update list widget
        self._refresh_saved_plots_list()

        self.statusBar().showMessage(f"Saved plot configuration: {plot_name}", 3000)

    def _save_plot_configs_to_hdf5(self):
        """Save all plot configurations to the HDF5 file as a JSON attribute."""
        if not self._current_csv_group_path or not self.model or not self.model.filepath:
            return

        try:

            with h5py.File(self.model.filepath, "r+") as h5:
                if self._current_csv_group_path in h5:
                    grp = h5[self._current_csv_group_path]
                    if isinstance(grp, h5py.Group):
                        if self._saved_plots:
                            # Convert plot configs to JSON string
                            plots_json = json.dumps(self._saved_plots)
                            grp.attrs["saved_plots"] = plots_json
                        else:
                            # Remove attribute if no plots
                            if "saved_plots" in grp.attrs:
                                del grp.attrs["saved_plots"]
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Warning: Could not save plot configs: {exc}", 5000)

    def _load_plot_configs_from_hdf5(self, grp: h5py.Group):
        """Load plot configurations from the HDF5 group attributes.

        Args:
            grp: HDF5 group to load plot configs from
        """
        try:
            if "saved_plots" in grp.attrs:

                plots_json = grp.attrs["saved_plots"]
                if isinstance(plots_json, bytes):
                    plots_json = plots_json.decode("utf-8")
                plots = json.loads(plots_json)
                # Validate format
                if isinstance(plots, list):
                    self._saved_plots = plots
                else:
                    self._saved_plots = []
            else:
                self._saved_plots = []
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not load plot configs from HDF5: {exc}")
            self._saved_plots = []

        # Refresh the list widget
        self._refresh_saved_plots_list()

    def _refresh_saved_plots_list(self):
        """Update the saved plots list widget with current configurations."""
        self.saved_plots_list.clear()
        for plot_config in self._saved_plots:
            name = plot_config.get("name", "Unnamed Plot")
            self.saved_plots_list.addItem(name)

        # Update button states
        self._update_plot_buttons_state()

    def _update_plot_buttons_state(self):
        """Enable/disable plot management buttons based on current state."""
        # Enable Save Plot button if CSV is loaded and columns are selected
        csv_loaded = self._current_csv_group_path is not None and self.preview_table.isVisible()
        sel_cols = self._get_selected_column_indices() if csv_loaded else []
        self.btn_save_plot.setEnabled(csv_loaded and len(sel_cols) >= 2)

        # Enable Delete and Edit Options buttons if a plot is selected
        has_selection = self.saved_plots_list.currentRow() >= 0
        self.btn_delete_plot.setEnabled(has_selection)
        self.btn_edit_plot_options.setEnabled(has_selection)

    def _on_saved_plot_selection_changed(self):
        """Handle selection change in saved plots list."""
        self._update_plot_buttons_state()

        # Automatically apply the selected plot
        current_item = self.saved_plots_list.currentItem()
        if current_item is not None:
            self._apply_saved_plot(current_item)

    def _apply_saved_plot(self, item=None):
        """Apply a saved plot configuration.

        Args:
            item: QListWidgetItem that was clicked/selected (optional)
        """
        if item is None:
            item = self.saved_plots_list.currentItem()

        if item is None:
            return

        # Get the plot configuration
        row = self.saved_plots_list.row(item)
        if row < 0 or row >= len(self._saved_plots):
            return

        plot_config = self._saved_plots[row]

        # Extract configuration
        x_idx = plot_config.get("x_col_idx")
        y_idxs = plot_config.get("y_col_idxs", [])
        filtered_indices = plot_config.get("filtered_indices")
        start_row = plot_config.get("start_row", 0)
        end_row = plot_config.get("end_row", -1)

        if x_idx is None or not y_idxs:
            QMessageBox.warning(
                self, "Invalid Configuration", "Plot configuration is missing column information."
            )
            return

        # Check if we have the CSV data loaded
        if not self._csv_data_dict or not self._current_csv_group_path:
            QMessageBox.information(self, "No Data", "CSV data is not loaded.")
            return

        # Get column names
        headers = [
            self.preview_table.horizontalHeaderItem(i).text()
            if self.preview_table.horizontalHeaderItem(i) is not None
            else f"col_{i}"
            for i in range(self.preview_table.columnCount())
        ]

        # Validate column indices
        if x_idx >= len(headers) or any(y_idx >= len(headers) for y_idx in y_idxs):
            QMessageBox.warning(
                self, "Invalid Columns", "Plot configuration references invalid column indices."
            )
            return

        try:
            x_name = headers[x_idx]
            y_names = [headers[i] for i in y_idxs]
        except Exception:
            QMessageBox.warning(
                self, "Plot Error", "Failed to resolve column headers for plotting."
            )
            return

        # Get the data with filtering applied
        col_data = {}
        for name in [x_name] + y_names:
            if name in self._csv_data_dict:
                full_data = self._csv_data_dict[name]

                if isinstance(full_data, np.ndarray):
                    # Use filtered_indices if available, otherwise fall back to start_row/end_row
                    if filtered_indices is not None:
                        # Use the stored filtered indices (handles non-contiguous filtering)
                        filtered_indices_array = np.array(filtered_indices, dtype=int)
                        col_data[name] = full_data[filtered_indices_array]
                    elif end_row >= 0 and end_row < len(full_data):
                        # Backward compatibility: use row range
                        col_data[name] = full_data[start_row : end_row + 1]
                    else:
                        col_data[name] = full_data[start_row:]
                else:
                    col_data[name] = np.array([full_data])

        if x_name not in col_data or not any(name in col_data for name in y_names):
            QMessageBox.warning(self, "Plot Error", "Failed to get column data for plotting.")
            return

        # Plot the data
        try:

            x_arr = col_data[x_name].ravel()
            min_len = min(len(x_arr), *(len(col_data.get(n, [])) for n in y_names if n in col_data))
            if min_len <= 0:
                QMessageBox.warning(self, "Plot Error", "No data to plot.")
                return

            # Get plot options from configuration
            plot_options = plot_config.get("plot_options", {})

            # Process x-axis data - check if it's datetime
            xaxis_datetime = plot_options.get("xaxis_datetime", False)
            datetime_format = plot_options.get("datetime_format", "").strip()

            # Check if x_arr contains strings (automatic date detection)
            x_is_string = False
            if len(x_arr) > 0:
                first_val = x_arr[0]
                x_is_string = isinstance(first_val, str) or (
                    hasattr(first_val, "dtype") and first_val.dtype.kind in ("U", "O")
                )

            # If x-axis is strings, try parsing as datetime
            if x_is_string and xaxis_datetime and not datetime_format:
                # Datetime mode enabled but no format specified - use auto-detection
                try:
                    # Try to parse as datetime without explicit format (pandas will infer)
                    x_data = pd.to_datetime(pd.Series(x_arr[:min_len]), errors="coerce")
                    # Check if parsing was successful (not all NaT/null)
                    valid_dates = x_data.notna()
                    if valid_dates.sum() > 0:
                        # Successfully parsed as dates - use datetime mode
                        # Convert to matplotlib date numbers using pandas
                        # Convert pandas datetime to numpy datetime64, then to matplotlib float dates
                        x_num = np.array([mdates.date2num(d) if pd.notna(d) else np.nan
                                         for d in x_data])
                        # Auto-detect worked, proceed with datetime
                    else:
                        # Parsing failed - treat as categorical/text
                        # Use integer indices for x-axis
                        x_num = np.arange(min_len, dtype=float)
                        xaxis_datetime = False
                except Exception:
                    # If auto-parsing fails, use integer indices
                    x_num = np.arange(min_len, dtype=float)
                    xaxis_datetime = False
            elif x_is_string and not xaxis_datetime:
                # Auto-detect dates even when checkbox not checked
                try:
                    # Try to parse as datetime without explicit format (pandas will infer)
                    x_data = pd.to_datetime(pd.Series(x_arr[:min_len]), errors="coerce")
                    # Check if parsing was successful (not all NaT/null)
                    valid_dates = x_data.notna()
                    if valid_dates.sum() > 0:
                        # Successfully parsed as dates - use datetime mode
                        # Convert to matplotlib date numbers using pandas
                        # Convert pandas datetime to numpy datetime64, then to matplotlib float dates
                        x_num = np.array([mdates.date2num(d) if pd.notna(d) else np.nan
                                         for d in x_data])
                        xaxis_datetime = True
                        # Auto-detect worked, proceed with datetime
                    else:
                        # Parsing failed - treat as categorical/text
                        # Use integer indices for x-axis
                        x_num = np.arange(min_len, dtype=float)
                        xaxis_datetime = False
                except Exception:
                    # If auto-parsing fails, use integer indices
                    x_num = np.arange(min_len, dtype=float)
                    xaxis_datetime = False
            elif xaxis_datetime and datetime_format:
                # Parse as datetime with explicit format
                try:
                    x_data = pd.to_datetime(
                        pd.Series(x_arr[:min_len]),
                        format=datetime_format,
                        errors="coerce"
                    )
                    # Convert to matplotlib dates
                    valid_dates = x_data.notna()
                    if valid_dates.sum() > 0:
                        # Convert pandas datetime to matplotlib numbers
                        x_num = np.array([mdates.date2num(d) if pd.notna(d) else np.nan
                                         for d in x_data])
                    else:
                        # No valid dates - fall back to numeric
                        x_num = (
                            pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce")
                            .astype(float)
                            .to_numpy()
                        )
                        xaxis_datetime = False
                except Exception:
                    # Fall back to numeric if datetime parsing fails
                    x_num = (
                        pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce")
                        .astype(float)
                        .to_numpy()
                    )
                    xaxis_datetime = False  # Disable datetime formatting
            else:
                # Process as numeric
                x_num = (
                    pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce")
                    .astype(float)
                    .to_numpy()
                )

            # Clear previous plot
            self.plot_figure.clear()
            ax = self.plot_figure.add_subplot(111)

            # Disable offset notation on axes
            ax.ticklabel_format(useOffset=False)

            # Get plot options from configuration
            plot_options = plot_config.get("plot_options", {})
            series_styles = plot_options.get("series", {})

            any_plotted = False
            for y_name in y_names:
                if y_name not in col_data:
                    continue
                y_arr = col_data[y_name].ravel()[:min_len]
                y_num = pd.to_numeric(pd.Series(y_arr), errors="coerce").astype(float).to_numpy()
                valid = np.isfinite(x_num) & np.isfinite(y_num)
                if valid.any():
                    # Get series-specific styling options
                    series_opts = series_styles.get(y_name, {})

                    # Use custom label if provided, otherwise use column name
                    label = series_opts.get("label", "").strip() or y_name

                    # Check if smoothing is enabled
                    apply_smooth = series_opts.get("smooth", False)
                    smooth_mode = series_opts.get("smooth_mode", "smoothed")
                    smooth_window = series_opts.get("smooth_window", 5)

                    # Plot original line if requested
                    if not apply_smooth or smooth_mode in ("original", "both"):
                        plot_kwargs = {"label": label if not (apply_smooth and smooth_mode == "both") else f"{label} (original)"}

                        if "color" in series_opts and series_opts["color"]:
                            plot_kwargs["color"] = series_opts["color"]
                        if "linestyle" in series_opts and series_opts["linestyle"]:
                            plot_kwargs["linestyle"] = series_opts["linestyle"]
                        if "marker" in series_opts and series_opts["marker"]:
                            plot_kwargs["marker"] = series_opts["marker"]
                        if "linewidth" in series_opts:
                            plot_kwargs["linewidth"] = series_opts["linewidth"]
                        if "markersize" in series_opts:
                            plot_kwargs["markersize"] = series_opts["markersize"]

                        # Make original line lighter/thinner if showing both
                        if apply_smooth and smooth_mode == "both":
                            plot_kwargs["alpha"] = 0.3
                            plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 1.5) * 0.7

                        ax.plot(x_num[valid], y_num[valid], **plot_kwargs)
                        any_plotted = True

                    # Plot smoothed line if requested
                    if apply_smooth and smooth_mode in ("smoothed", "both"):
                        try:
                            # Apply moving average smoothing
                            window = max(2, int(smooth_window))
                            # Use pandas rolling mean for smoothing
                            y_series = pd.Series(y_num[valid])
                            y_smooth = y_series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

                            smooth_kwargs = {"label": f"{label} (MA-{window})" if smooth_mode == "both" else label}

                            if "color" in series_opts and series_opts["color"]:
                                smooth_kwargs["color"] = series_opts["color"]
                            if "linestyle" in series_opts and series_opts["linestyle"]:
                                smooth_kwargs["linestyle"] = series_opts["linestyle"]
                            else:
                                smooth_kwargs["linestyle"] = "-"  # Default to solid for smoothed
                            if "marker" in series_opts and series_opts["marker"] and smooth_mode != "both":
                                smooth_kwargs["marker"] = series_opts["marker"]
                            if "linewidth" in series_opts:
                                smooth_kwargs["linewidth"] = series_opts["linewidth"]
                            else:
                                smooth_kwargs["linewidth"] = 2.0  # Slightly thicker for smoothed
                            if "markersize" in series_opts and smooth_mode != "both":
                                smooth_kwargs["markersize"] = series_opts["markersize"]

                            ax.plot(x_num[valid], y_smooth, **smooth_kwargs)
                            any_plotted = True
                        except Exception as e:
                            # If smoothing fails, fall back to original data
                            print(f"Smoothing failed for {y_name}: {e}")
                            if not (smooth_mode == "both"):  # Only plot if we haven't already
                                plot_kwargs = {"label": label}
                                if "color" in series_opts and series_opts["color"]:
                                    plot_kwargs["color"] = series_opts["color"]
                                if "linestyle" in series_opts and series_opts["linestyle"]:
                                    plot_kwargs["linestyle"] = series_opts["linestyle"]
                                if "marker" in series_opts and series_opts["marker"]:
                                    plot_kwargs["marker"] = series_opts["marker"]
                                if "linewidth" in series_opts:
                                    plot_kwargs["linewidth"] = series_opts["linewidth"]
                                if "markersize" in series_opts:
                                    plot_kwargs["markersize"] = series_opts["markersize"]
                                ax.plot(x_num[valid], y_num[valid], **plot_kwargs)
                                any_plotted = True

                    # Plot trend line if requested
                    apply_trend = series_opts.get("trendline", False)
                    if apply_trend:
                        try:
                            trend_type = series_opts.get("trendline_type", "linear")
                            trend_mode = series_opts.get("trendline_mode", "both")

                            # Calculate trend line using numpy polyfit
                            if trend_type == "linear":
                                degree = 1
                            elif trend_type == "poly2":
                                degree = 2
                            elif trend_type == "poly3":
                                degree = 3
                            elif trend_type == "poly4":
                                degree = 4
                            else:
                                degree = 1

                            # Fit polynomial to the data
                            coeffs = np.polyfit(x_num[valid], y_num[valid], degree)
                            poly = np.poly1d(coeffs)
                            y_trend = poly(x_num[valid])

                            # Prepare trend line label
                            if degree == 1:
                                trend_label = f"{label} (linear trend)"
                            else:
                                trend_label = f"{label} (poly{degree} trend)"

                            # Plot trend line
                            trend_kwargs = {"label": trend_label}
                            if "color" in series_opts and series_opts["color"]:
                                trend_kwargs["color"] = series_opts["color"]
                            trend_kwargs["linestyle"] = "--"  # Dashed for trend lines
                            trend_kwargs["linewidth"] = 2.0
                            trend_kwargs["alpha"] = 0.8

                            ax.plot(x_num[valid], y_trend, **trend_kwargs)
                            any_plotted = True
                        except Exception as e:
                            # If trend line calculation fails, silently continue
                            print(f"Trend line calculation failed for {y_name}: {e}")

            if not any_plotted:
                QMessageBox.information(self, "Plot", "No valid numeric data found to plot.")
                return

            # Apply custom labels or use defaults
            xlabel = plot_options.get("xlabel", "").strip() or x_name
            ylabel = plot_options.get("ylabel", "").strip() or ", ".join(y_names)

            # Set title with plot name and row range info
            custom_title = plot_options.get("title", "").strip()
            if custom_title:
                title = custom_title
            else:
                title = plot_config.get("name", "Plot")
                if start_row > 0 or end_row < len(self._csv_data_dict.get(x_name, [])) - 1:
                    title += f" (rows {start_row}-{end_row})"

            # Apply font family if specified
            font_family = plot_options.get("font_family", "serif")

            # Apply labels and title with font sizes and family
            ax.set_title(title, fontsize=plot_options.get("title_fontsize", 12), family=font_family)
            ax.set_xlabel(xlabel, fontsize=plot_options.get("axis_label_fontsize", 10), family=font_family)
            ax.set_ylabel(ylabel, fontsize=plot_options.get("axis_label_fontsize", 10), family=font_family)
            ax.tick_params(axis='both', which='major', labelsize=plot_options.get("tick_fontsize", 9))
            # Apply font family to tick labels
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily(font_family)

            # Apply grid and legend options
            if plot_options.get("grid", True):
                ax.grid(True)
            if plot_options.get("legend", True):
                legend = ax.legend(fontsize=plot_options.get("legend_fontsize", 9))
                # Apply font family to legend text
                for text in legend.get_texts():
                    text.set_fontfamily(font_family)

            # Format datetime x-axis if enabled
            if xaxis_datetime:
                datetime_display_format = plot_options.get("datetime_display_format", "").strip()
                if datetime_display_format:
                    # Use custom format
                    ax.xaxis.set_major_formatter(DateFormatter(datetime_display_format))
                else:
                    # Use automatic date formatting
                    ax.xaxis.set_major_locator(AutoDateLocator())
                # Rotate labels for better readability
                self.plot_figure.autofmt_xdate()
            elif x_is_string and not xaxis_datetime:
                # X-axis is categorical strings - set string labels on integer positions
                # Limit labels to avoid overcrowding
                num_points = min(min_len, len(x_arr))
                if num_points <= 50:
                    # Show all labels if not too many
                    ax.set_xticks(np.arange(num_points))
                    ax.set_xticklabels(x_arr[:num_points], rotation=45, ha="right")
                else:
                    # Show subset of labels to avoid overcrowding
                    step = max(1, num_points // 20)  # Show ~20 labels
                    indices = np.arange(0, num_points, step)
                    ax.set_xticks(indices)
                    ax.set_xticklabels([x_arr[i] for i in indices], rotation=45, ha="right")

            # Apply axis limits if specified
            xlim_min = plot_options.get("xlim_min")
            xlim_max = plot_options.get("xlim_max")
            if xlim_min is not None or xlim_max is not None:
                current_xlim = ax.get_xlim()
                new_xlim = (
                    xlim_min if xlim_min is not None else current_xlim[0],
                    xlim_max if xlim_max is not None else current_xlim[1],
                )
                ax.set_xlim(new_xlim)

            ylim_min = plot_options.get("ylim_min")
            ylim_max = plot_options.get("ylim_max")
            if ylim_min is not None or ylim_max is not None:
                current_ylim = ax.get_ylim()
                new_ylim = (
                    ylim_min if ylim_min is not None else current_ylim[0],
                    ylim_max if ylim_max is not None else current_ylim[1],
                )
                ax.set_ylim(new_ylim)

            # Apply log scale if requested
            if plot_options.get("xlog", False):
                ax.set_xscale("log")
            if plot_options.get("ylog", False):
                ax.set_yscale("log")

            # Draw reference lines
            reference_lines = plot_options.get("reference_lines", [])
            for refline in reference_lines:
                try:
                    line_type = refline.get("type")
                    value = refline.get("value")
                    color = refline.get("color", "black")
                    linestyle = refline.get("linestyle", "solid")
                    linewidth = refline.get("linewidth", 1.5)
                    label = refline.get("label")

                    if line_type == "horizontal" and value is not None:
                        ax.axhline(y=value, color=color, linestyle=linestyle,
                                   linewidth=linewidth, label=label)
                    elif line_type == "vertical" and value is not None:
                        ax.axvline(x=value, color=color, linestyle=linestyle,
                                   linewidth=linewidth, label=label)
                except Exception:
                    # Skip invalid reference lines
                    pass

            self.plot_figure.tight_layout()

            # Refresh canvas
            self.plot_canvas.draw()

            # Switch to Plot tab
            self.bottom_tabs.setCurrentIndex(1)

            self.statusBar().showMessage(
                f"Applied plot: {plot_config.get('name', 'Unnamed')}", 3000
            )

        except Exception as exc:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot data:\n{exc}")

    def _export_plot_to_file(self, plot_config, filepath):
        """Export a plot configuration to a file.

        Args:
            plot_config: Plot configuration dictionary
            filepath: Target file path for export

        Returns:
            tuple: (success: bool, error_msg: str) - True and empty string if successful, False and error message otherwise
        """
        try:

            # Get group path and column names from plot config
            # Support backward compatibility: use current CSV group if not in config
            group_path = plot_config.get("csv_group_path")
            if not group_path:
                # Fallback to current CSV group path (for older configs)
                group_path = self._current_csv_group_path
                if not group_path:
                    return False, "No CSV group path available (load CSV data first)"

            x_idx = plot_config.get("x_col_idx")
            y_idxs = plot_config.get("y_col_idxs", [])

            # Get column names from the config or current table
            stored_columns = plot_config.get("column_names", [])
            if stored_columns:
                headers = stored_columns
            else:
                # Fallback to current table headers (for older configs)
                headers = [
                    self.preview_table.horizontalHeaderItem(i).text()
                    if self.preview_table.horizontalHeaderItem(i) is not None
                    else f"col_{i}"
                    for i in range(self.preview_table.columnCount())
                ]

            if x_idx >= len(headers) or not all(idx < len(headers) for idx in y_idxs):
                return False, "Invalid column indices"

            x_name = headers[x_idx]
            y_names = [headers[i] for i in y_idxs]

            # Read column data
            col_data = self._read_csv_columns(group_path, headers)
            if not col_data:
                return False, "Failed to read column data"

            if x_name not in col_data or not any(name in col_data for name in y_names):
                return False, "Column data not found"

            # Apply filtering if specified in plot config
            # This ensures filtered data is used during export (respects filters from when plot was saved)
            filtered_indices = plot_config.get("filtered_indices")

            if filtered_indices is not None:
                # Use the stored filtered indices array (handles non-contiguous filtering)
                filtered_indices_array = np.array(filtered_indices, dtype=int)
                filtered_col_data = {}
                for col_name, col_array in col_data.items():
                    if isinstance(col_array, np.ndarray) and len(col_array) > 0:
                        # Apply filtering using the saved indices
                        filtered_col_data[col_name] = col_array[filtered_indices_array]
                    else:
                        filtered_col_data[col_name] = col_array
                col_data = filtered_col_data
            else:
                # Backward compatibility: use start_row/end_row if filtered_indices not available
                start_row = plot_config.get("start_row", 0)
                end_row = plot_config.get("end_row")

                if end_row is not None and end_row >= start_row:
                    # Filter all columns to the saved row range
                    filtered_col_data = {}
                    for col_name, col_array in col_data.items():
                        if isinstance(col_array, np.ndarray) and len(col_array) > 0:
                            # Apply slice [start_row:end_row+1] to get inclusive range
                            actual_end = min(end_row + 1, len(col_array))
                            filtered_col_data[col_name] = col_array[start_row:actual_end]
                        else:
                            filtered_col_data[col_name] = col_array
                    col_data = filtered_col_data

            # Get plot options
            plot_options = plot_config.get("plot_options", {})
            figwidth = plot_options.get("figwidth", 8.0)
            figheight = plot_options.get("figheight", 6.0)
            dpi = plot_options.get("dpi", 100)

            # Create figure with specified size
            fig = Figure(figsize=(figwidth, figheight), dpi=dpi)
            ax = fig.add_subplot(111)

            # Disable offset notation on axes
            ax.ticklabel_format(useOffset=False)

            # Process x-axis data (same logic as _apply_saved_plot)
            x_arr = col_data[x_name].ravel()
            min_len = min(len(x_arr), *(len(col_data.get(n, [])) for n in y_names if n in col_data))
            if min_len <= 0:
                return False, "No data to plot"

            xaxis_datetime = plot_options.get("xaxis_datetime", False)
            datetime_format = plot_options.get("datetime_format", "").strip()

            # Check if x_arr contains strings
            x_is_string = False
            if len(x_arr) > 0:
                first_val = x_arr[0]
                x_is_string = isinstance(first_val, str) or (
                    hasattr(first_val, "dtype") and first_val.dtype.kind in ("U", "O")
                )

            # Date parsing logic (same as _apply_saved_plot)
            if x_is_string and xaxis_datetime and not datetime_format:
                try:
                    x_data = pd.to_datetime(pd.Series(x_arr[:min_len]), errors="coerce")
                    valid_dates = x_data.notna()
                    if valid_dates.sum() > 0:
                        x_num = np.array([mdates.date2num(d) if pd.notna(d) else np.nan for d in x_data])
                    else:
                        x_num = np.arange(min_len, dtype=float)
                        xaxis_datetime = False
                except Exception:
                    x_num = np.arange(min_len, dtype=float)
                    xaxis_datetime = False
            elif x_is_string and not xaxis_datetime:
                try:
                    x_data = pd.to_datetime(pd.Series(x_arr[:min_len]), errors="coerce")
                    valid_dates = x_data.notna()
                    if valid_dates.sum() > 0:
                        x_num = np.array([mdates.date2num(d) if pd.notna(d) else np.nan for d in x_data])
                        xaxis_datetime = True
                    else:
                        x_num = np.arange(min_len, dtype=float)
                        xaxis_datetime = False
                except Exception:
                    x_num = np.arange(min_len, dtype=float)
                    xaxis_datetime = False
            elif xaxis_datetime and datetime_format:
                try:
                    x_data = pd.to_datetime(pd.Series(x_arr[:min_len]), format=datetime_format, errors="coerce")
                    valid_dates = x_data.notna()
                    if valid_dates.sum() > 0:
                        x_num = np.array([mdates.date2num(d) if pd.notna(d) else np.nan for d in x_data])
                    else:
                        x_num = pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce").astype(float).to_numpy()
                        xaxis_datetime = False
                except Exception:
                    x_num = pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce").astype(float).to_numpy()
                    xaxis_datetime = False
            else:
                x_num = pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce").astype(float).to_numpy()

            # Plot series with smoothing support
            series_styles = plot_options.get("series", {})
            any_plotted = False

            for y_name in y_names:
                if y_name not in col_data:
                    continue
                y_arr = col_data[y_name].ravel()[:min_len]
                y_num = pd.to_numeric(pd.Series(y_arr), errors="coerce").astype(float).to_numpy()
                valid = np.isfinite(x_num) & np.isfinite(y_num)
                if valid.any():
                    series_opts = series_styles.get(y_name, {})
                    label = series_opts.get("label", "").strip() or y_name

                    # Smoothing logic
                    apply_smooth = series_opts.get("smooth", False)
                    smooth_mode = series_opts.get("smooth_mode", "smoothed")
                    smooth_window = series_opts.get("smooth_window", 5)

                    # Plot original if requested
                    if not apply_smooth or smooth_mode in ("original", "both"):
                        plot_kwargs = {"label": label if not (apply_smooth and smooth_mode == "both") else f"{label} (original)"}
                        if "color" in series_opts and series_opts["color"]:
                            plot_kwargs["color"] = series_opts["color"]
                        if "linestyle" in series_opts and series_opts["linestyle"]:
                            plot_kwargs["linestyle"] = series_opts["linestyle"]
                        if "marker" in series_opts and series_opts["marker"]:
                            plot_kwargs["marker"] = series_opts["marker"]
                        if "linewidth" in series_opts:
                            plot_kwargs["linewidth"] = series_opts["linewidth"]
                        if "markersize" in series_opts:
                            plot_kwargs["markersize"] = series_opts["markersize"]

                        if apply_smooth and smooth_mode == "both":
                            plot_kwargs["alpha"] = 0.3
                            plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 1.5) * 0.7

                        ax.plot(x_num[valid], y_num[valid], **plot_kwargs)
                        any_plotted = True

                    # Plot smoothed if requested
                    if apply_smooth and smooth_mode in ("smoothed", "both"):
                        try:
                            window = max(2, int(smooth_window))
                            y_series = pd.Series(y_num[valid])
                            y_smooth = y_series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

                            smooth_kwargs = {"label": f"{label} (MA-{window})" if smooth_mode == "both" else label}
                            if "color" in series_opts and series_opts["color"]:
                                smooth_kwargs["color"] = series_opts["color"]
                            if "linestyle" in series_opts and series_opts["linestyle"]:
                                smooth_kwargs["linestyle"] = series_opts["linestyle"]
                            else:
                                smooth_kwargs["linestyle"] = "-"
                            if "linewidth" in series_opts:
                                smooth_kwargs["linewidth"] = series_opts["linewidth"]
                            else:
                                smooth_kwargs["linewidth"] = 2.0

                            ax.plot(x_num[valid], y_smooth, **smooth_kwargs)
                            any_plotted = True
                        except Exception:
                            # Smoothing failed, already plotted original if needed
                            pass

                    # Plot trend line if requested
                    apply_trend = series_opts.get("trendline", False)
                    if apply_trend:
                        try:
                            trend_type = series_opts.get("trendline_type", "linear")

                            # Calculate trend line using numpy polyfit
                            if trend_type == "linear":
                                degree = 1
                            elif trend_type == "poly2":
                                degree = 2
                            elif trend_type == "poly3":
                                degree = 3
                            elif trend_type == "poly4":
                                degree = 4
                            else:
                                degree = 1

                            # Fit polynomial to the data
                            coeffs = np.polyfit(x_num[valid], y_num[valid], degree)
                            poly = np.poly1d(coeffs)
                            y_trend = poly(x_num[valid])

                            # Prepare trend line label
                            if degree == 1:
                                trend_label = f"{label} (linear trend)"
                            else:
                                trend_label = f"{label} (poly{degree} trend)"

                            # Plot trend line
                            trend_kwargs = {"label": trend_label}
                            if "color" in series_opts and series_opts["color"]:
                                trend_kwargs["color"] = series_opts["color"]
                            trend_kwargs["linestyle"] = "--"
                            trend_kwargs["linewidth"] = 2.0
                            trend_kwargs["alpha"] = 0.8

                            ax.plot(x_num[valid], y_trend, **trend_kwargs)
                            any_plotted = True
                        except Exception:
                            # Trend line calculation failed, silently continue
                            pass

            if not any_plotted:
                return False, "No valid numeric data to plot"

            # Apply labels and formatting
            xlabel = plot_options.get("xlabel", "").strip() or x_name
            ylabel = plot_options.get("ylabel", "").strip() or ", ".join(y_names)
            custom_title = plot_options.get("title", "").strip()
            title_text = custom_title if custom_title else plot_config.get("name", "Plot")

            # Apply font family if specified
            font_family = plot_options.get("font_family", "serif")

            # Apply labels with font sizes and family
            ax.set_title(title_text, fontsize=plot_options.get("title_fontsize", 12), family=font_family)
            ax.set_xlabel(xlabel, fontsize=plot_options.get("axis_label_fontsize", 10), family=font_family)
            ax.set_ylabel(ylabel, fontsize=plot_options.get("axis_label_fontsize", 10), family=font_family)
            ax.tick_params(axis='both', which='major', labelsize=plot_options.get("tick_fontsize", 9))
            # Apply font family to tick labels
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily(font_family)

            # Apply grid and legend
            if plot_options.get("grid", True):
                ax.grid(True, alpha=0.3)
            if plot_options.get("legend", True):
                legend = ax.legend(fontsize=plot_options.get("legend_fontsize", 9))
                # Apply font family to legend text
                for text in legend.get_texts():
                    text.set_fontfamily(font_family)

            # Apply axis limits
            xlim_min = plot_options.get("xlim_min")
            xlim_max = plot_options.get("xlim_max")
            if xlim_min is not None or xlim_max is not None:
                current_xlim = ax.get_xlim()
                new_xlim = (
                    xlim_min if xlim_min is not None else current_xlim[0],
                    xlim_max if xlim_max is not None else current_xlim[1]
                )
                ax.set_xlim(new_xlim)

            ylim_min = plot_options.get("ylim_min")
            ylim_max = plot_options.get("ylim_max")
            if ylim_min is not None or ylim_max is not None:
                current_ylim = ax.get_ylim()
                new_ylim = (
                    ylim_min if ylim_min is not None else current_ylim[0],
                    ylim_max if ylim_max is not None else current_ylim[1]
                )
                ax.set_ylim(new_ylim)

            # Apply log scale
            if plot_options.get("xlog", False):
                ax.set_xscale("log")
            if plot_options.get("ylog", False):
                ax.set_yscale("log")

            # Date formatting
            if xaxis_datetime:
                display_format = plot_options.get("datetime_display_format", "").strip()
                if display_format:
                    ax.xaxis.set_major_formatter(DateFormatter(display_format))
                else:
                    ax.xaxis.set_major_locator(AutoDateLocator())
                fig.autofmt_xdate()

            # Reference lines
            ref_lines = plot_options.get("reference_lines", [])
            for refline in ref_lines:
                try:
                    if refline.get("type") == "horizontal":
                        ax.axhline(
                            y=refline.get("value", 0),
                            color=refline.get("color", "red"),
                            linestyle=refline.get("linestyle", "--"),
                            linewidth=refline.get("linewidth", 1.0),
                            label=refline.get("label")
                        )
                    elif refline.get("type") == "vertical":
                        ax.axvline(
                            x=refline.get("value", 0),
                            color=refline.get("color", "red"),
                            linestyle=refline.get("linestyle", "--"),
                            linewidth=refline.get("linewidth", 1.0),
                            label=refline.get("label")
                        )
                except Exception:
                    pass

            fig.tight_layout()

            # Save to file
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

            return True, ""

        except Exception as e:
            error_msg = f"Error exporting plot: {e}"
            print(error_msg)
            traceback.print_exc()
            return False, str(e)

    def _delete_plot_config(self):
        """Delete the selected plot configuration."""
        current_row = self.saved_plots_list.currentRow()
        if current_row < 0 or current_row >= len(self._saved_plots):
            return

        plot_config = self._saved_plots[current_row]
        plot_name = plot_config.get("name", "Unnamed Plot")

        # Confirm deletion
        resp = QMessageBox.question(
            self,
            "Delete Plot Configuration",
            f"Are you sure you want to delete '{plot_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if resp != QMessageBox.Yes:
            return

        # Remove from list
        del self._saved_plots[current_row]

        # Save to HDF5
        self._save_plot_configs_to_hdf5()

        # Refresh list widget
        self._refresh_saved_plots_list()

        self.statusBar().showMessage(f"Deleted plot configuration: {plot_name}", 3000)

    def _edit_plot_options_dialog(self):
        """Open dialog to edit plot options for the selected plot configuration."""
        current_row = self.saved_plots_list.currentRow()
        if current_row < 0 or current_row >= len(self._saved_plots):
            return

        plot_config = self._saved_plots[current_row]

        # Get column names from the preview table
        headers = [
            self.preview_table.horizontalHeaderItem(i).text()
            if self.preview_table.horizontalHeaderItem(i) is not None
            else f"col_{i}"
            for i in range(self.preview_table.columnCount())
        ]

        # Show the options dialog (pass all headers so indices work correctly)
        dialog = PlotOptionsDialog(plot_config, headers, self)
        if dialog.exec() == QDialog.Accepted:
            # Update the configuration with the new options
            updated_config = dialog.get_plot_config()
            self._saved_plots[current_row] = updated_config

            # Save to HDF5
            self._save_plot_configs_to_hdf5()

            # Refresh the list (in case the name changed)
            self._refresh_saved_plots_list()

            # Re-select the same row
            self.saved_plots_list.setCurrentRow(current_row)

            # Reapply the plot to show the changes
            self._apply_saved_plot(None)

            self.statusBar().showMessage(
                f"Updated plot options: {updated_config.get('name', 'Unnamed')}", 3000
            )

    def _on_saved_plots_context_menu(self, point):
        """Show context menu for saved plots list."""
        item = self.saved_plots_list.itemAt(point)
        if item is None:
            return

        menu = QMenu(self)
        act_delete = menu.addAction("Delete Plot")

        global_pos = self.saved_plots_list.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)

        if chosen == act_delete:
            self._delete_plot_config()


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
