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
from qtpy.QtCore import QMimeData, QSettings, QSize, QUrl, Qt
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
    QListWidgetItem,
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
    QStyle,
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
        """Initialize the draggable list widget.

        Args:
            parent: Parent widget (should be the HDF5Viewer instance)
        """
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.parent_viewer = parent

    def mimeData(self, items):
        """Create mime data for drag operation.

        Args:
            items: List of QListWidgetItem objects being dragged

        Returns:
            QMimeData object for the drag operation
        """
        mime_data = super().mimeData(items)
        if items and self.parent_viewer:
            # Store the row index in the mime data
            row = self.row(items[0])
            mime_data.setText(str(row))
        return mime_data

    def startDrag(self, supportedActions):
        """Start drag operation and export plot to temporary file.

        Args:
            supportedActions: Qt.DropActions flags indicating supported drop actions
        """
        current_row = self.currentRow()
        if current_row < 0 or not self.parent_viewer:
            return

        # Export the plot to a temporary file
        try:
            # Get plot configuration and make a copy to capture current UI state
            plot_config = self.parent_viewer._saved_plots[current_row].copy()

            # Capture current visibility state from the displayed plot
            series_visibility = self.parent_viewer._capture_plot_visibility_state()

            # Update plot_config with current visibility state
            plot_options = plot_config.get("plot_options", {}).copy()
            plot_options["series_visibility"] = series_visibility
            plot_config["plot_options"] = plot_options

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
        """Initialize the resizable label.

        Args:
            parent: Parent widget
            rescale_callback: Callback function to execute on resize events
        """
        super().__init__(parent)
        self._rescale_callback = rescale_callback

    def resizeEvent(self, event):
        """Handle resize events and trigger rescale callback.

        Args:
            event: QResizeEvent object
        """
        if self._rescale_callback:
            self._rescale_callback()
        super().resizeEvent(event)


class DropTreeView(QTreeView):
    """TreeView that accepts external file/folder drops and forwards to the viewer."""

    def __init__(self, parent=None, viewer=None):
        """Initialize the tree view with drag-and-drop support.

        Args:
            parent: Parent widget
            viewer: HDF5Viewer instance to forward drag-and-drop operations to
        """
        super().__init__(parent)
        self.viewer = viewer
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """Handle drag enter events for file/folder drops.

        Args:
            event: QDragEnterEvent object
        """
        md = event.mimeData()
        if md and md.hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        """Handle drag move events for file/folder drops.

        Args:
            event: QDragMoveEvent object
        """
        md = event.mimeData()
        if md and md.hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        """Handle drop events for file/folder drops and internal HDF5 moves.

        Args:
            event: QDropEvent object
        """
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
                    self.viewer._update_file_size_display()
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
                self.viewer._update_file_size_display()
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
        """Initialize the column statistics dialog.

        Args:
            column_names: List of column names to display statistics for
            data_dict: Dictionary mapping column names to data arrays
            filtered_indices: Array of filtered row indices, or None for all rows
            parent: Parent widget
        """
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
        """Initialize the column sorting dialog.

        Args:
            column_names: List of available column names for sorting
            parent: Parent widget
        """
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
        add_btn.setToolTip("Add a new column to sort by (columns are sorted in order from top to bottom)")
        add_btn.clicked.connect(self._add_sort_row)
        layout.addWidget(add_btn)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _add_sort_row(self, col_name=None, ascending=True):
        """Add a new sort row to the dialog.

        Args:
            col_name: Initial column name to select, or None for first column
            ascending: True for ascending sort, False for descending
        """
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
        up_btn.setToolTip("Move this sort column up (higher priority)")
        up_btn.setMaximumWidth(30)
        up_btn.clicked.connect(lambda: self._move_sort_row(row_widget, -1))

        # Move down button
        down_btn = QPushButton("↓")
        down_btn.setToolTip("Move this sort column down (lower priority)")
        down_btn.setMaximumWidth(30)
        down_btn.clicked.connect(lambda: self._move_sort_row(row_widget, 1))

        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Remove this sort column")
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
        """Move a sort row up or down in the list.

        Args:
            row_widget: The widget representing the sort row to move
            direction: -1 to move up, +1 to move down
        """
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
        """Remove a sort row from the dialog.

        Args:
            row_widget: The widget representing the sort row to remove
        """
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
        """Set the sort specifications to display in the dialog.

        Args:
            sort_specs: List of (column_name, ascending) tuples
        """
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
        add_btn.setToolTip("Add a new filter condition (all filters are combined with AND logic)")
        add_btn.clicked.connect(self._add_filter_row)
        layout.addWidget(add_btn)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _add_filter_row(self, col_name=None, operator="==", value=""):
        """Add a new filter row to the dialog.

        Args:
            col_name: Initial column name to filter on, or None for first column
            operator: Comparison operator (==, !=, >, >=, <, <=, contains, startswith, endswith)
            value: Filter value as string
        """
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
        remove_btn.setToolTip("Remove this filter condition")
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
        """Remove a filter row from the dialog.

        Args:
            row_widget: The widget representing the filter row to remove
        """
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
        """Set initial filters from a list of (column_name, operator, value) tuples.

        Args:
            filters: List of (column_name, operator, value) tuples
        """
        for col_name, operator, value in filters:
            self._add_filter_row(col_name, operator, value)


class ColumnVisibilityDialog(QDialog):
    """Dialog for selecting which columns to display in the CSV table."""

    def __init__(self, column_names, visible_columns=None, parent=None):
        """Initialize the column visibility dialog.

        Args:
            column_names: List of all column names
            visible_columns: List of currently visible column names, or None for all
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Select Columns to Display")
        self.resize(400, 500)

        self.column_names = column_names
        self.visible_columns = visible_columns if visible_columns is not None else column_names.copy()

        layout = QVBoxLayout(self)

        # Instructions
        info_label = QLabel("Select which columns to display in the table:")
        layout.addWidget(info_label)

        # Show all / Select specific radio buttons
        radio_layout = QHBoxLayout()
        self.radio_show_all = QCheckBox("Show All Columns")
        self.radio_show_all.setChecked(len(self.visible_columns) == len(self.column_names))
        self.radio_show_all.toggled.connect(self._on_show_all_toggled)
        radio_layout.addWidget(self.radio_show_all)
        radio_layout.addStretch()
        layout.addLayout(radio_layout)

        # Column list with checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.StyledPanel)

        list_container = QWidget()
        self.list_layout = QVBoxLayout(list_container)
        self.list_layout.setContentsMargins(5, 5, 5, 5)

        self.column_checkboxes = []
        for col_name in self.column_names:
            checkbox = QCheckBox(col_name)
            checkbox.setChecked(col_name in self.visible_columns)
            checkbox.toggled.connect(self._on_checkbox_toggled)
            self.column_checkboxes.append(checkbox)
            self.list_layout.addWidget(checkbox)

        self.list_layout.addStretch()
        scroll.setWidget(list_container)
        layout.addWidget(scroll)

        # Select/Deselect buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        button_layout.addWidget(deselect_all_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_show_all_toggled(self, checked):
        """Handle show all checkbox toggle.

        Args:
            checked: True if checkbox is checked, False otherwise
        """
        if checked:
            for checkbox in self.column_checkboxes:
                checkbox.setChecked(True)

    def _on_checkbox_toggled(self):
        """Update show all checkbox when individual checkboxes change."""
        all_checked = all(cb.isChecked() for cb in self.column_checkboxes)
        self.radio_show_all.setChecked(all_checked)

    def _select_all(self):
        """Select all columns."""
        for checkbox in self.column_checkboxes:
            checkbox.setChecked(True)

    def _deselect_all(self):
        """Deselect all columns."""
        for checkbox in self.column_checkboxes:
            checkbox.setChecked(False)

    def get_visible_columns(self):
        """Return list of selected column names."""
        return [cb.text() for cb in self.column_checkboxes if cb.isChecked()]


class UniqueValuesDialog(QDialog):
    """Dialog for displaying unique values in a column."""

    def __init__(self, column_name, unique_values, parent=None):
        """Initialize the unique values dialog.

        Args:
            column_name: Name of the column being displayed
            unique_values: List of unique values to display
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(f"Unique Values - {column_name}")
        self.resize(400, 500)

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(f"Unique values in column '{column_name}' ({len(unique_values)} unique):")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Table to display unique values
        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Populate table with unique values
        self.table.setRowCount(len(unique_values))
        for i, value in enumerate(unique_values):
            item = QTableWidgetItem(str(value))
            self.table.setItem(i, 0, item)

        layout.addWidget(self.table)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)


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
        """Initialize the plot options dialog.

        Args:
            plot_config: Dictionary containing plot configuration
            column_names: List of available column names
            parent: Parent widget
        """
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

        grid_layout.addWidget(QLabel("Legend Position:"))
        self.legend_loc_combo = QComboBox()
        # Matplotlib legend location options
        legend_locations = [
            ("best", "Best"),
            ("upper right", "Upper Right"),
            ("upper left", "Upper Left"),
            ("lower left", "Lower Left"),
            ("lower right", "Lower Right"),
            ("right", "Right"),
            ("center left", "Center Left"),
            ("center right", "Center Right"),
            ("lower center", "Lower Center"),
            ("upper center", "Upper Center"),
            ("center", "Center"),
        ]
        for loc_value, loc_name in legend_locations:
            self.legend_loc_combo.addItem(loc_name, loc_value)
        # Set current value
        current_loc = self.plot_config.get("plot_options", {}).get("legend_loc", "best")
        index = self.legend_loc_combo.findData(current_loc)
        if index >= 0:
            self.legend_loc_combo.setCurrentIndex(index)
        grid_layout.addWidget(self.legend_loc_combo)

        self.dark_background_checkbox = QCheckBox("Dark Background")
        self.dark_background_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("dark_background", False)
        )
        grid_layout.addWidget(self.dark_background_checkbox)

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
        # Ensure y_idxs is a list (handle legacy configs where it might be an integer)
        if isinstance(y_idxs, int):
            y_idxs = [y_idxs]
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
        add_hline_btn.setToolTip("Add a horizontal reference line at a specific Y value")
        add_hline_btn.clicked.connect(lambda: self._add_refline_widget("horizontal"))
        reflines_buttons.addWidget(add_hline_btn)

        add_vline_btn = QPushButton("+ Add Vertical Line")
        add_vline_btn.setToolTip("Add a vertical reference line at a specific X value")
        add_vline_btn.clicked.connect(lambda: self._add_refline_widget("vertical"))
        reflines_buttons.addWidget(add_vline_btn)

        reflines_buttons.addStretch()
        reflines_layout.addLayout(reflines_buttons)

        tabs.addTab(reflines_tab, "Reference Lines")

        # Tab 6: Filters & Sort
        filters_sort_tab = QWidget()
        filters_sort_layout = QVBoxLayout(filters_sort_tab)

        # Filters section
        filters_label = QLabel("<b>Data Filters:</b>")
        filters_sort_layout.addWidget(filters_label)

        filters_info = QLabel("Configure which rows from the CSV are included in this plot.")
        filters_info.setWordWrap(True)
        filters_sort_layout.addWidget(filters_info)

        # Store filters and sort in dialog - use list() to create proper copies
        self.csv_filters = list(self.plot_config.get("csv_filters", []))
        self.csv_sort = list(self.plot_config.get("csv_sort", []))

        # Filter status display
        self.filter_status_label = QLabel(self._get_filter_status_text())
        self.filter_status_label.setWordWrap(True)
        filters_sort_layout.addWidget(self.filter_status_label)

        # Filter buttons
        filter_buttons_layout = QHBoxLayout()
        edit_filters_btn = QPushButton("Configure Filters...")
        edit_filters_btn.setToolTip("Add or modify filter conditions")
        edit_filters_btn.clicked.connect(self._edit_filters)
        filter_buttons_layout.addWidget(edit_filters_btn)

        clear_filters_btn = QPushButton("Clear Filters")
        clear_filters_btn.setToolTip("Remove all filter conditions")
        clear_filters_btn.clicked.connect(self._clear_filters)
        filter_buttons_layout.addWidget(clear_filters_btn)

        filter_buttons_layout.addStretch()
        filters_sort_layout.addLayout(filter_buttons_layout)

        # Sort section
        filters_sort_layout.addSpacing(20)
        sort_label = QLabel("<b>Data Sorting:</b>")
        filters_sort_layout.addWidget(sort_label)

        sort_info = QLabel("Configure how the data is sorted before plotting.")
        sort_info.setWordWrap(True)
        filters_sort_layout.addWidget(sort_info)

        # Sort status display
        self.sort_status_label = QLabel(self._get_sort_status_text())
        self.sort_status_label.setWordWrap(True)
        filters_sort_layout.addWidget(self.sort_status_label)

        # Sort buttons
        sort_buttons_layout = QHBoxLayout()
        edit_sort_btn = QPushButton("Configure Sort...")
        edit_sort_btn.setToolTip("Configure multi-column sorting")
        edit_sort_btn.clicked.connect(self._edit_sort)
        sort_buttons_layout.addWidget(edit_sort_btn)

        clear_sort_btn = QPushButton("Clear Sort")
        clear_sort_btn.setToolTip("Remove all sorting")
        clear_sort_btn.clicked.connect(self._clear_sort)
        sort_buttons_layout.addWidget(clear_sort_btn)

        sort_buttons_layout.addStretch()
        filters_sort_layout.addLayout(sort_buttons_layout)

        filters_sort_layout.addStretch()
        tabs.addTab(filters_sort_tab, "Data")

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_series_widget(self, series_name, series_idx, series_options):
        """Create a widget for configuring one series.

        Args:
            series_name: Name of the data series (column name)
            series_idx: Index of the series in the list
            series_options: Dictionary of existing options for this series

        Returns:
            QFrame widget configured for the series
        """
        widget = QFrame()
        widget.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Series name label
        name_label = QLabel(f"<b>{series_name}</b>")
        layout.addWidget(name_label)

        # Plot type selection
        plot_type_layout = QHBoxLayout()
        plot_type_layout.addWidget(QLabel("Plot Type:"))
        plot_type_combo = QComboBox()
        plot_type_combo.addItem("Line", "line")
        plot_type_combo.addItem("Bar", "bar")
        current_plot_type = series_options.get("plot_type", "line")
        plot_type_idx = 0 if current_plot_type == "line" else 1
        plot_type_combo.setCurrentIndex(plot_type_idx)
        plot_type_combo.setToolTip("Choose between line plot or bar chart")
        plot_type_combo.setMinimumWidth(100)
        plot_type_layout.addWidget(plot_type_combo)
        plot_type_layout.addStretch()
        layout.addLayout(plot_type_layout)

        # Color, line style, and marker in one row
        style_layout = QHBoxLayout()

        # Color picker button
        style_layout.addWidget(QLabel("Color:"))
        color_button = QPushButton()
        color_button.setToolTip("Click to choose a custom color for this data series")
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
        widget._plot_type_combo = plot_type_combo
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
        remove_btn.setToolTip("Remove this reference line from the plot")
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
        color_button.setToolTip("Click to choose the color for this reference line")
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
        """Remove a reference line widget.

        Args:
            widget: The reference line widget to remove
        """
        if widget in self.refline_widgets:
            self.refline_widgets.remove(widget)
            self.reflines_layout.removeWidget(widget)
            widget.deleteLater()

    def _get_filter_status_text(self):
        """Get status text for filters."""
        if not self.csv_filters:
            return "No filters applied"
        return f"{len(self.csv_filters)} filter(s) applied: " + ", ".join(
            f"{col} {op} {val}" for col, op, val in self.csv_filters[:3]
        ) + ("..." if len(self.csv_filters) > 3 else "")

    def _get_sort_status_text(self):
        """Get status text for sort."""
        if not self.csv_sort:
            return "No sorting applied"
        return f"Sorted by {len(self.csv_sort)} column(s): " + ", ".join(
            f"{col} ({order})" for col, order in self.csv_sort[:3]
        ) + ("..." if len(self.csv_sort) > 3 else "")

    def _edit_filters(self):
        """Open the filter configuration dialog."""
        dialog = ColumnFilterDialog(self.column_names, self)
        dialog.set_filters(self.csv_filters)

        if dialog.exec() == QDialog.Accepted:
            self.csv_filters = dialog.get_filters()
            self.filter_status_label.setText(self._get_filter_status_text())

    def _clear_filters(self):
        """Clear all filters."""
        self.csv_filters = []
        self.filter_status_label.setText(self._get_filter_status_text())

    def _edit_sort(self):
        """Open the sort configuration dialog."""
        dialog = ColumnSortDialog(self.column_names, self)
        dialog.set_sort_specs(self.csv_sort)

        if dialog.exec() == QDialog.Accepted:
            self.csv_sort = dialog.get_sort_specs()
            self.sort_status_label.setText(self._get_sort_status_text())

    def _clear_sort(self):
        """Clear all sorting."""
        self.csv_sort = []
        self.sort_status_label.setText(self._get_sort_status_text())

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
        plot_opts["legend_loc"] = self.legend_loc_combo.currentData()
        plot_opts["dark_background"] = self.dark_background_checkbox.isChecked()

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
                "plot_type": widget._plot_type_combo.currentData(),
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

        # Save filters and sort
        self.plot_config["csv_filters"] = self.csv_filters
        self.plot_config["csv_sort"] = self.csv_sort

        return self.plot_config


class CustomSplitter(QSplitter):
    """QSplitter with explicit cursor management for macOS compatibility."""

    def __init__(self, *args, **kwargs):
        """Initialize the clickable splitter.

        Args:
            *args: Positional arguments passed to QSplitter
            **kwargs: Keyword arguments passed to QSplitter
        """
        super().__init__(*args, **kwargs)
        self._setupCursor()

    def _setupCursor(self):
        """Ensure the splitter handle always has the correct cursor based on orientation."""
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
        """Override to set cursor on handle after widget is added.

        Args:
            widget: QWidget to add to the splitter
        """
        super().addWidget(widget)
        self._setupCursor()

    def insertWidget(self, index, widget):
        """Override to set cursor on handle after widget is inserted.

        Args:
            index: Position to insert the widget at
            widget: QWidget to insert
        """
        super().insertWidget(index, widget)
        self._setupCursor()


class HDF5Viewer(QMainWindow):
    def __init__(self, parent=None):
        """Initialize the HDF5 viewer main window.

        Args:
            parent: Parent widget, or None for top-level window
        """
        super().__init__(parent)
        self.setWindowTitle("VibeHDF5")
        self.resize(900, 600)
        self._original_pixmap = None
        self._current_highlighter = None  # Track current syntax highlighter

        # Initialize QSettings for storing recent files
        self.settings = QSettings("VibeHDF5", "VibeHDF5")
        self.max_recent_files = 10
        self.recent_file_actions = []

        # Central widget: splitter with tree (left) and preview (right)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        splitter = CustomSplitter(self)
        splitter.setHandleWidth(8)  # Make handle wider for easier grabbing
        splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely
        splitter.splitterMoved.connect(self._on_splitter_moved)
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

        # Create a vertical splitter for tree view and saved plots list
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setHandleWidth(8)
        left_splitter.setChildrenCollapsible(False)

        # Tree view widget
        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)
        tree_layout.setContentsMargins(0, 0, 0, 0)

        self.tree = DropTreeView(tree_widget, viewer=self)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionBehavior(QTreeView.SelectRows)
        self.tree.setHeaderHidden(False)
        tree_layout.addWidget(self.tree)

        left_splitter.addWidget(tree_widget)

        # Saved plots widget (below tree)
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)

        saved_plots_label = QLabel("Saved Plots:")
        saved_plots_label.setStyleSheet("font-weight: bold; padding: 4px;")
        plots_layout.addWidget(saved_plots_label)

        self.saved_plots_list = DraggablePlotListWidget(self)
        self.saved_plots_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.saved_plots_list.customContextMenuRequested.connect(self._on_saved_plots_context_menu)
        self.saved_plots_list.setToolTip("Drag and drop to filesystem to export plot. Double-click to rename.")
        self.saved_plots_list.setEditTriggers(QListWidget.DoubleClicked)
        plots_layout.addWidget(self.saved_plots_list)

        # Buttons for plot management
        plot_buttons_layout = QHBoxLayout()
        self.btn_save_plot = QPushButton("Save Plot")
        self.btn_save_plot.setToolTip("Save current table column selection as a named plot configuration")
        self.btn_save_plot.clicked.connect(self._save_plot_config_dialog)
        self.btn_save_plot.setEnabled(False)
        plot_buttons_layout.addWidget(self.btn_save_plot)

        self.btn_edit_plot_options = QPushButton("Edit Options")
        self.btn_edit_plot_options.setToolTip("Customize appearance, styling, and export settings for the selected plot")
        self.btn_edit_plot_options.clicked.connect(self._edit_plot_options_dialog)
        self.btn_edit_plot_options.setEnabled(False)
        plot_buttons_layout.addWidget(self.btn_edit_plot_options)

        self.btn_delete_plot = QPushButton("Delete")
        self.btn_delete_plot.setToolTip("Delete the selected plot configuration permanently")
        self.btn_delete_plot.clicked.connect(self._delete_plot_config)
        self.btn_delete_plot.setEnabled(False)
        plot_buttons_layout.addWidget(self.btn_delete_plot)

        plots_layout.addLayout(plot_buttons_layout)

        left_splitter.addWidget(plots_widget)

        # Set initial sizes for the splitter (tree gets more space)
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 1)

        left_layout.addWidget(left_splitter)

        splitter.addWidget(left)

        self.model = HDF5TreeModel(self)
        self.tree.setModel(self.model)
        self.tree.header().setStretchLastSection(True)
        self.tree.header().setDefaultSectionSize(350)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)

        # Enable editing via double-click for renaming
        self.tree.setEditTriggers(QAbstractItemView.DoubleClicked)

        # Enable drag-and-drop (both export and external import)
        self.tree.setDragEnabled(True)  # allow dragging out
        self.tree.setAcceptDrops(True)  # allow external drops
        self.tree.setDragDropMode(QAbstractItemView.DragDrop)
        self.tree.setDefaultDropAction(Qt.MoveAction)  # move for internal, copy for external

        # Context menu on tree
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.on_tree_context_menu)

        # Connect to model data changes to handle renames
        self.model.dataChanged.connect(self._on_tree_item_renamed)

        # Preview panel (right)
        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Create a vertical splitter for content and attributes
        right_splitter = CustomSplitter(Qt.Vertical, right)
        right_splitter.setHandleWidth(8)  # Make handle wider for easier grabbing
        right_splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely
        right_splitter.splitterMoved.connect(self._on_splitter_moved)
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
        self.btn_configure_filters.setToolTip("Add or modify filter conditions to show only specific rows (filters are saved with the file)")
        self.btn_configure_filters.clicked.connect(self._configure_filters_dialog)
        filter_panel_layout.addWidget(self.btn_configure_filters)

        self.btn_clear_filters = QPushButton("Clear Filters")
        self.btn_clear_filters.setToolTip("Remove all active filters and show all rows")
        self.btn_clear_filters.clicked.connect(self._clear_filters)
        self.btn_clear_filters.setEnabled(False)
        filter_panel_layout.addWidget(self.btn_clear_filters)

        self.btn_show_statistics = QPushButton("Statistics...")
        self.btn_show_statistics.setToolTip("View statistical summaries (min, max, mean, median, etc.) for each column using filtered data")
        self.btn_show_statistics.clicked.connect(self._show_statistics_dialog)
        filter_panel_layout.addWidget(self.btn_show_statistics)

        self.btn_configure_sort = QPushButton("Sort...")
        self.btn_configure_sort.setToolTip("Configure multi-column sorting with ascending/descending order (sort settings are saved with the file)")
        self.btn_configure_sort.clicked.connect(self._configure_sort_dialog)
        filter_panel_layout.addWidget(self.btn_configure_sort)

        self.btn_clear_sort = QPushButton("Clear Sort")
        self.btn_clear_sort.setToolTip("Remove all sorting and display rows in original order")
        self.btn_clear_sort.clicked.connect(self._clear_sort)
        self.btn_clear_sort.setEnabled(False)
        filter_panel_layout.addWidget(self.btn_clear_sort)

        self.btn_configure_columns = QPushButton("Columns...")
        self.btn_configure_columns.setToolTip("Select which columns to display in the table")
        self.btn_configure_columns.clicked.connect(self._configure_columns_dialog)
        filter_panel_layout.addWidget(self.btn_configure_columns)

        filter_panel_layout.addStretch()

        self.filter_panel.setVisible(False)
        content_layout.addWidget(self.filter_panel)

        # Table widget for CSV/tabular data (hidden by default)
        self.preview_table = QTableWidget(self)
        self.preview_table.setVisible(False)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(False)
        # Enable selecting multiple columns for plotting
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.preview_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.preview_table.itemSelectionChanged.connect(self._update_plot_action_enabled)
        # Connect scrollbar for lazy loading
        self.preview_table.verticalScrollBar().valueChanged.connect(self._on_table_scroll)
        # Enable context menu on horizontal header
        self.preview_table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.preview_table.horizontalHeader().customContextMenuRequested.connect(self._on_column_header_context_menu)
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
        # Enable context menu for attributes table
        self.attrs_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.attrs_table.customContextMenuRequested.connect(self._on_attrs_context_menu)
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
        self._create_menu_bar()
        self.setStatusBar(QStatusBar(self))

        # Add permanent file size label to right side of status bar
        self.file_size_label = QLabel()
        self.file_size_label.setStyleSheet("QLabel { padding: 0 5px; }")
        self.statusBar().addPermanentWidget(self.file_size_label)

        self.tree.selectionModel().selectionChanged.connect(self.on_selection_changed)

        # Track currently previewed CSV group (for plotting)
        self._current_csv_group_path: str | None = None

        # Track CSV data and filters
        self._csv_data_dict: dict[str, np.ndarray] = {}  # Cached column data (lazy loaded)
        self._csv_dataset_info: dict = {}  # Metadata for lazy loading (column -> (ds_key, th_grp_path, row_count, dtype))
        self._csv_column_names: list[str] = []
        self._csv_filters: list[tuple[str, str, str]] = []  # (column, operator, value)
        self._csv_filtered_indices: np.ndarray | None = None  # Indices of visible rows
        self._csv_visible_columns: list[str] = []  # Columns to show in table

        # Track saved plot configurations for current CSV group
        self._saved_plots: list[dict] = []  # List of plot config dictionaries

        # Connect saved plots list selection changed
        self.saved_plots_list.itemSelectionChanged.connect(self._on_saved_plot_selection_changed)
        self.saved_plots_list.itemClicked.connect(self._on_saved_plot_clicked)
        self.saved_plots_list.itemChanged.connect(self._on_plot_item_renamed)

        # Track current search pattern
        self._search_pattern: str = ""

    def _create_progress_dialog(self, title: str, max_value: int = 100, min_duration: int = 500) -> QProgressDialog:
        """Create a standard progress dialog with consistent settings.

        Args:
            title: Dialog title/label text
            max_value: Maximum progress value (default 100)
            min_duration: Minimum duration in ms before showing (default 500)

        Returns:
            Configured QProgressDialog instance
        """
        progress = QProgressDialog(title, "Cancel", 0, max_value, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(min_duration)
        progress.setValue(0)
        QApplication.processEvents()
        return progress

    def _save_csv_attr_to_hdf5(self, attr_name: str, value, success_msg: str, clear_msg: str = None):
        """Generic helper to save a CSV group attribute to HDF5.

        Args:
            attr_name: Name of the attribute to save
            value: Value to save (will be JSON encoded if not empty)
            success_msg: Status bar message on successful save
            clear_msg: Status bar message when clearing attribute (optional)
        """
        if not self._current_csv_group_path or not self.model or not self.model.filepath:
            return

        try:
            with h5py.File(self.model.filepath, "r+") as h5:
                if self._current_csv_group_path in h5:
                    grp = h5[self._current_csv_group_path]
                    if isinstance(grp, h5py.Group):
                        if value:
                            # Save the value as JSON
                            json_str = json.dumps(value)
                            grp.attrs[attr_name] = json_str
                            self.statusBar().showMessage(success_msg, 3000)
                        else:
                            # Remove attribute if value is empty
                            if attr_name in grp.attrs:
                                del grp.attrs[attr_name]
                            if clear_msg:
                                self.statusBar().showMessage(clear_msg, 3000)

            # Update file size display after modification
            self._update_file_size_display()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not save {attr_name} to HDF5: {exc}")

    def _load_csv_attr_from_hdf5(self, grp: h5py.Group, attr_name: str, validator=None):
        """Generic helper to load a CSV group attribute from HDF5.

        Args:
            grp: HDF5 group to load from
            attr_name: Name of the attribute to load
            validator: Optional function to validate/transform loaded value

        Returns:
            Loaded value or None if not found/invalid
        """
        try:
            if attr_name in grp.attrs:
                json_str = grp.attrs[attr_name]
                if isinstance(json_str, bytes):
                    json_str = json_str.decode("utf-8")
                value = json.loads(json_str)
                if validator:
                    return validator(value)
                return value
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not load {attr_name} from HDF5: {exc}")
        return None

    def _apply_plot_style(self, fig_or_ax, ax, use_dark: bool) -> None:
        """Apply dark or light background styling to a plot figure and axes.

        Args:
            fig_or_ax: Figure object (for setting figure background)
            ax: Axes object (for setting axes colors)
            use_dark: True for dark background, False for light background
        """
        if use_dark:
            fig_or_ax.set_facecolor('#1e1e1e')
            ax.set_facecolor('#2e2e2e')
            color = 'white'
        else:
            fig_or_ax.set_facecolor('white')
            ax.set_facecolor('white')
            color = 'black'

        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(color)

        # Set label and tick colors
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis='x', colors=color)
        ax.tick_params(axis='y', colors=color)

    def _parse_datetime_column(self, x_arr: np.ndarray, min_len: int,
                                datetime_format: str = "") -> tuple[np.ndarray, bool]:
        """Parse a column as datetime and convert to matplotlib date numbers.

        Args:
            x_arr: Array of values to parse as dates
            min_len: Maximum length to process
            datetime_format: Optional strptime format string

        Returns:
            Tuple of (numeric_array, success_flag)
        """
        try:
            if datetime_format:
                x_data = pd.to_datetime(pd.Series(x_arr[:min_len]),
                                       format=datetime_format, errors="coerce")
            else:
                x_data = pd.to_datetime(pd.Series(x_arr[:min_len]), errors="coerce")

            valid_dates = x_data.notna()
            if valid_dates.sum() > 0:
                x_num = np.array([mdates.date2num(d) if pd.notna(d) else np.nan
                                 for d in x_data])
                return x_num, True
        except Exception:
            pass

        return np.arange(min_len, dtype=float), False

    def _make_legend_interactive(self, ax: plt.Axes, legend) -> None:
        """Make the legend interactive so clicking on labels toggles line visibility.

        Args:
            ax: Matplotlib axes object
            legend: Legend object to make interactive
        """
        if not legend:
            return

        # Get the figure from the axes
        fig = ax.get_figure()
        if not fig:
            return

        # Map legend lines to plot lines
        line_dict = {}
        for legend_line, orig_line in zip(legend.get_lines(), ax.get_lines()):
            legend_line.set_picker(True)
            legend_line.set_pickradius(5)
            line_dict[legend_line] = orig_line

        # Connect pick event
        def on_pick(event):
            legend_line = event.artist
            if legend_line not in line_dict:
                return

            orig_line = line_dict[legend_line]
            visible = not orig_line.get_visible()
            orig_line.set_visible(visible)

            # Update legend line appearance
            if visible:
                legend_line.set_alpha(1.0)
            else:
                legend_line.set_alpha(0.2)

            # Use the figure's canvas to redraw
            if fig.canvas:
                fig.canvas.draw_idle()

        # Store the connection ID to avoid multiple connections
        if not hasattr(self, '_legend_pick_cid'):
            self._legend_pick_cid = None

        # Disconnect previous connection if exists
        if self._legend_pick_cid is not None:
            try:
                self.plot_figure.canvas.mpl_disconnect(self._legend_pick_cid)
            except Exception:
                pass

        # Connect new pick event to the current figure
        self._legend_pick_cid = fig.canvas.mpl_connect('pick_event', on_pick)

    def _capture_plot_visibility_state(self) -> dict[str, bool]:
        """Capture current visibility state of all plot lines.

        Returns:
            Dictionary mapping series label to visibility state
        """
        series_visibility = {}
        if hasattr(self, 'plot_figure') and self.plot_figure.axes:
            ax = self.plot_figure.axes[0]
            for line in ax.get_lines():
                label = line.get_label()
                if label and not label.startswith('_'):  # Ignore internal matplotlib labels
                    series_visibility[label] = line.get_visible()
        return series_visibility

    def _apply_plot_visibility_state(self, ax: plt.Axes, y_names: list[str], series_visibility: dict[str, bool]) -> None:
        """Apply visibility state to plot lines and update legend appearance.

        Args:
            ax: Matplotlib axes object
            y_names: List of Y column names
            series_visibility: Dictionary mapping series label to visibility state
        """
        # Apply visibility to plot lines
        for line in ax.get_lines():
            label = line.get_label()
            if label in series_visibility:
                line.set_visible(series_visibility[label])

        # Update legend appearance to reflect hidden series
        if series_visibility and ax.get_legend():
            legend = ax.get_legend()
            for legend_line, orig_line in zip(legend.get_lines(), ax.get_lines()):
                if not orig_line.get_visible():
                    legend_line.set_alpha(0.2)

    def _process_x_axis_data(self, x_idx: int | None, col_data: dict, y_names: list[str],
                            x_name: str, plot_options: dict) -> tuple[np.ndarray, np.ndarray, bool, bool, int]:
        """Process X-axis data for plotting, handling single-column, datetime, and string data.

        Args:
            x_idx: Column index for X axis, or None for point count
            col_data: Dictionary of column name -> array data
            y_names: List of Y column names
            x_name: X column name (or "Point" if x_idx is None)
            plot_options: Dictionary of plot options

        Returns:
            Tuple of (x_arr, x_num, x_is_string, xaxis_datetime, min_len)
        """
        # Calculate minimum data length
        min_len = min(len(col_data.get(n, [])) for n in y_names if n in col_data)

        # Handle single-column mode (x_idx is None)
        if x_idx is None:
            x_arr = np.arange(min_len)
            x_num = x_arr.astype(float)
            return x_arr, x_num, False, False, min_len

        # Multi-column mode - process X data
        x_arr = col_data[x_name].ravel()
        min_len = min(len(x_arr), min_len)

        xaxis_datetime = plot_options.get("xaxis_datetime", False)
        datetime_format = plot_options.get("datetime_format", "").strip()

        # Check if x_arr contains strings
        x_is_string = False
        if len(x_arr) > 0:
            first_val = x_arr[0]
            x_is_string = isinstance(first_val, str) or (
                hasattr(first_val, "dtype") and first_val.dtype.kind in ("U", "O")
            )

        # Parse X data based on type
        if x_is_string:
            # Try datetime parsing with various strategies
            if xaxis_datetime and not datetime_format:
                # Auto-detect datetime format
                x_num, success = self._parse_datetime_column(x_arr, min_len)
                if success:
                    xaxis_datetime = True
                else:
                    x_num = np.arange(min_len, dtype=float)
                    xaxis_datetime = False
            elif not xaxis_datetime:
                # Auto-detect datetime even if not explicitly requested
                x_num, success = self._parse_datetime_column(x_arr, min_len)
                xaxis_datetime = success
            elif xaxis_datetime and datetime_format:
                # Use specified format
                x_num, success = self._parse_datetime_column(x_arr, min_len, datetime_format)
                if not success:
                    # Fall back to numeric conversion
                    x_num = pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce").astype(float).to_numpy()
                    xaxis_datetime = False
            else:
                x_num = np.arange(min_len, dtype=float)
        else:
            # Non-string data - convert to numeric
            x_num = pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce").astype(float).to_numpy()
            xaxis_datetime = False

        return x_arr, x_num, x_is_string, xaxis_datetime, min_len

    def _format_xaxis(self, ax, fig, xaxis_datetime: bool, x_is_string: bool,
                      x_arr: np.ndarray, min_len: int, plot_options: dict | None = None) -> None:
        """Format x-axis for datetime or categorical string data.

        Args:
            ax: Matplotlib axes object
            fig: Matplotlib figure object (for autofmt_xdate)
            xaxis_datetime: True if x-axis contains datetime data
            x_is_string: True if x-axis contains string data
            x_arr: Original x-axis array (for string labels)
            min_len: Length of data
            plot_options: Optional plot options dict (for custom datetime format)
        """
        if xaxis_datetime:
            # Format datetime x-axis
            if plot_options:
                datetime_display_format = plot_options.get("datetime_display_format", "").strip()
                if datetime_display_format:
                    ax.xaxis.set_major_formatter(DateFormatter(datetime_display_format))
                else:
                    ax.xaxis.set_major_locator(AutoDateLocator())
            else:
                ax.xaxis.set_major_locator(AutoDateLocator())
            # Rotate labels for better readability
            fig.autofmt_xdate()
        elif x_is_string and not xaxis_datetime:
            # X-axis is categorical strings - set string labels on integer positions
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

    def _apply_axis_limits(self, ax, plot_options: dict) -> None:
        """Apply axis limit settings from plot options.

        Args:
            ax: Matplotlib axes object
            plot_options: Plot options dict containing xlim/ylim settings
        """
        # Apply x-axis limits
        xlim_min = plot_options.get("xlim_min")
        xlim_max = plot_options.get("xlim_max")
        if xlim_min is not None or xlim_max is not None:
            current_xlim = ax.get_xlim()
            new_xlim = (
                xlim_min if xlim_min is not None else current_xlim[0],
                xlim_max if xlim_max is not None else current_xlim[1],
            )
            ax.set_xlim(new_xlim)

        # Apply y-axis limits
        ylim_min = plot_options.get("ylim_min")
        ylim_max = plot_options.get("ylim_max")
        if ylim_min is not None or ylim_max is not None:
            current_ylim = ax.get_ylim()
            new_ylim = (
                ylim_min if ylim_min is not None else current_ylim[0],
                ylim_max if ylim_max is not None else current_ylim[1],
            )
            ax.set_ylim(new_ylim)

    def _plot_series_with_options(
        self,
        ax: plt.Axes,
        x_num: np.ndarray,
        y_num: np.ndarray,
        valid: np.ndarray,
        y_name: str,
        series_opts: dict,
        any_plotted: bool,
    ) -> bool:
        """Plot a single series with smoothing and trendline options applied.

        Args:
            ax: Matplotlib axes to plot on
            x_num: X-axis data (numeric)
            y_num: Y-axis data (numeric)
            valid: Boolean mask for valid data points
            y_name: Name of the series
            series_opts: Dictionary of series options (color, smoothing, trendlines, etc.)
            any_plotted: Whether any data has been plotted yet

        Returns:
            Updated any_plotted flag
        """
        label = series_opts.get("label", "").strip() or y_name
        plot_type = series_opts.get("plot_type", "line")

        # Smoothing logic
        apply_smooth = series_opts.get("smooth", False)
        smooth_mode = series_opts.get("smooth_mode", "smoothed")
        smooth_window = series_opts.get("smooth_window", 5)

        # Plot original if requested
        if not apply_smooth or smooth_mode in ("original", "both"):
            plot_kwargs = {
                "label": label if not (apply_smooth and smooth_mode == "both") else f"{label} (original)"
            }
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

            # Use bar or plot depending on plot_type
            if plot_type == "bar":
                # For bar charts, remove line/marker specific options
                bar_kwargs = {"label": plot_kwargs.get("label")}
                if "color" in plot_kwargs:
                    bar_kwargs["color"] = plot_kwargs["color"]
                if "alpha" in plot_kwargs:
                    bar_kwargs["alpha"] = plot_kwargs["alpha"]
                ax.bar(x_num[valid], y_num[valid], **bar_kwargs)
            else:
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

                # Use bar or plot depending on plot_type
                if plot_type == "bar":
                    bar_kwargs = {"label": smooth_kwargs.get("label")}
                    if "color" in smooth_kwargs:
                        bar_kwargs["color"] = smooth_kwargs["color"]
                    ax.bar(x_num[valid], y_smooth, **bar_kwargs)
                else:
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

        return any_plotted

    def _apply_plot_labels_and_formatting(
        self,
        ax: plt.Axes,
        fig,
        x_name: str,
        y_names: list,
        plot_config: dict,
        plot_options: dict,
        use_dark: bool,
        interactive_legend: bool = True,
    ) -> None:
        """Apply labels, fonts, grid, legend, log scale, and reference lines to a plot.

        Args:
            ax: Matplotlib axes object
            fig: Matplotlib figure object
            x_name: Name of x-axis data
            y_names: List of y-axis data names
            plot_config: Plot configuration dictionary
            plot_options: Plot options dictionary
            use_dark: Whether dark background is enabled
            interactive_legend: Whether to make legend interactive (only for UI plots)
        """
        # Apply custom labels or use defaults
        xlabel = plot_options.get("xlabel", "").strip() or x_name
        ylabel = plot_options.get("ylabel", "").strip() or ", ".join(y_names)
        custom_title = plot_options.get("title", "").strip()
        title_text = custom_title if custom_title else plot_config.get("name", "Plot")

        # Apply font family if specified
        font_family = plot_options.get("font_family", "serif")

        # Apply labels with font sizes and family
        title_obj = ax.set_title(title_text, fontsize=plot_options.get("title_fontsize", 12), family=font_family)
        title_obj.set_color('white' if use_dark else 'black')
        ax.set_xlabel(xlabel, fontsize=plot_options.get("axis_label_fontsize", 10), family=font_family)
        ax.set_ylabel(ylabel, fontsize=plot_options.get("axis_label_fontsize", 10), family=font_family)
        ax.tick_params(axis='both', which='major', labelsize=plot_options.get("tick_fontsize", 9))

        # Apply font family to tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)

        # Apply grid and legend
        grid_alpha = 0.3 if plot_options.get("grid", True) else None
        if grid_alpha:
            ax.grid(True, alpha=grid_alpha)
        if plot_options.get("legend", True):
            legend_loc = plot_options.get("legend_loc", "best")
            legend = ax.legend(
                fontsize=plot_options.get("legend_fontsize", 9),
                loc=legend_loc
            )
            # Apply font family to legend text
            for text in legend.get_texts():
                text.set_fontfamily(font_family)
            # Make legend interactive (only for UI plots, not exports)
            if interactive_legend:
                self._make_legend_interactive(ax, legend)

        # Apply log scale
        if plot_options.get("xlog", False):
            ax.set_xscale("log")
        if plot_options.get("ylog", False):
            ax.set_yscale("log")

        # Reference lines
        ref_lines = plot_options.get("reference_lines", [])
        for refline in ref_lines:
            try:
                line_type = refline.get("type")
                value = refline.get("value")
                if line_type == "horizontal" and value is not None:
                    ax.axhline(
                        y=value,
                        color=refline.get("color", "red"),
                        linestyle=refline.get("linestyle", "--"),
                        linewidth=refline.get("linewidth", 1.0),
                        label=refline.get("label")
                    )
                elif line_type == "vertical" and value is not None:
                    ax.axvline(
                        x=value,
                        color=refline.get("color", "red"),
                        linestyle=refline.get("linestyle", "--"),
                        linewidth=refline.get("linewidth", 1.0),
                        label=refline.get("label")
                    )
            except Exception:
                pass

        fig.tight_layout()

    def _create_actions(self) -> None:
        """Create all QAction objects for menu and toolbar items."""
        # Get standard icon theme
        style = self.style()

        self.act_new = QAction("New HDF5 File…", self)
        self.act_new.setIcon(style.standardIcon(QStyle.SP_FileIcon))
        self.act_new.setShortcut("Ctrl+N")
        self.act_new.setToolTip("Create a new empty HDF5 file (Ctrl+N)")
        self.act_new.triggered.connect(self.new_file_dialog)

        self.act_open = QAction("Open HDF5…", self)
        self.act_open.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.setToolTip("Open an existing HDF5 file for browsing and editing (Ctrl+O)")
        self.act_open.triggered.connect(self.open_file_dialog)

        # Add files/folder actions
        self.act_add_files = QAction("Add Files…", self)
        self.act_add_files.setIcon(style.standardIcon(QStyle.SP_FileDialogNewFolder))
        self.act_add_files.setShortcut("Ctrl+Shift+F")
        self.act_add_files.setToolTip("Import one or more files into the HDF5 archive (Ctrl+Shift+F)")
        self.act_add_files.triggered.connect(self.add_files_dialog)

        self.act_add_folder = QAction("Add Folder…", self)
        self.act_add_folder.setIcon(style.standardIcon(QStyle.SP_FileDialogNewFolder))
        self.act_add_folder.setShortcut("Ctrl+Shift+D")
        self.act_add_folder.setToolTip("Import an entire folder structure recursively into the HDF5 archive (Ctrl+Shift+D)")
        self.act_add_folder.triggered.connect(self.add_folder_dialog)

        self.act_new_folder = QAction("New Folder…", self)
        self.act_new_folder.setIcon(style.standardIcon(QStyle.SP_FileDialogNewFolder))
        self.act_new_folder.setShortcut("Ctrl+Shift+N")
        self.act_new_folder.setToolTip("Create a new empty group (folder) in the selected location (Ctrl+Shift+N)")
        self.act_new_folder.triggered.connect(self.new_folder_dialog)

        self.act_expand = QAction("Expand All", self)
        self.act_expand.setIcon(style.standardIcon(QStyle.SP_TitleBarUnshadeButton))
        self.act_expand.setToolTip("Expand all groups in the tree view to show full hierarchy")
        self.act_expand.triggered.connect(self.tree.expandAll)

        self.act_collapse = QAction("Collapse All", self)
        self.act_collapse.setIcon(style.standardIcon(QStyle.SP_TitleBarShadeButton))
        self.act_collapse.setToolTip("Collapse all groups in the tree view to show only top level")
        self.act_collapse.triggered.connect(self.tree.collapseAll)

        self.act_quit = QAction("Quit", self)
        self.act_quit.setIcon(style.standardIcon(QStyle.SP_DialogCloseButton))
        self.act_quit.setShortcut("Ctrl+Q")
        self.act_quit.setToolTip("Close the application (Ctrl+Q)")
        self.act_quit.triggered.connect(self.close)

        # About action
        self.act_about = QAction("About VibeHDF5", self)
        self.act_about.setIcon(style.standardIcon(QStyle.SP_MessageBoxInformation))
        self.act_about.setToolTip("Show information about VibeHDF5")
        self.act_about.triggered.connect(self.show_about_dialog)

        # Plotting action for CSV tables
        self.act_plot_selected = QAction("Plot Selected Columns", self)
        self.act_plot_selected.setIcon(style.standardIcon(QStyle.SP_FileDialogContentsView))
        self.act_plot_selected.setToolTip(
            "Plot selected table columns (1 column: Y vs point count; 2+ columns: first is X, others are Y)"
        )
        self.act_plot_selected.triggered.connect(self.plot_selected_columns)
        self.act_plot_selected.setEnabled(False)

        # Font size actions
        self.act_increase_font = QAction("Increase Font Size", self)
        self.act_increase_font.setShortcut("Ctrl++")
        self.act_increase_font.setToolTip("Increase GUI font size (Ctrl++)")
        self.act_increase_font.triggered.connect(self._increase_font_size)

        self.act_decrease_font = QAction("Decrease Font Size", self)
        self.act_decrease_font.setShortcut("Ctrl+-")
        self.act_decrease_font.setToolTip("Decrease GUI font size (Ctrl+-)")
        self.act_decrease_font.triggered.connect(self._decrease_font_size)

        self.act_reset_font = QAction("Reset Font Size", self)
        self.act_reset_font.setShortcut("Ctrl+0")
        self.act_reset_font.setToolTip("Reset GUI font size to default (Ctrl+0)")
        self.act_reset_font.triggered.connect(self._reset_font_size)

        # Repack file action
        self.act_repack = QAction("Repack File...", self)
        self.act_repack.setToolTip("Reclaim space from deleted items by repacking the HDF5 file")
        self.act_repack.triggered.connect(self._repack_file_dialog)

        # File properties action
        self.act_file_properties = QAction("File Properties...", self)
        self.act_file_properties.setToolTip("View detailed information about the HDF5 file")
        self.act_file_properties.triggered.connect(self._show_file_properties_dialog)

        # Merge file action
        self.act_merge_file = QAction("Merge File...", self)
        self.act_merge_file.setToolTip("Import contents from another HDF5 file into the current file")
        self.act_merge_file.triggered.connect(self._merge_file_dialog)

        # Create recent file actions (will be populated dynamically)
        for i in range(self.max_recent_files):
            action = QAction(self)
            action.setVisible(False)
            action.triggered.connect(self._open_recent_file)
            self.recent_file_actions.append(action)

    def _create_toolbar(self) -> None:
        """Create and populate the main toolbar."""
        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(16, 16))  # Smaller icon size
        self.addToolBar(tb)
        tb.addAction(self.act_new)
        tb.addAction(self.act_open)
        tb.addSeparator()
        tb.addAction(self.act_plot_selected)

    def _create_menu_bar(self) -> None:
        """Create and populate the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.act_new)
        file_menu.addAction(self.act_open)
        file_menu.addSeparator()

        # Recent files submenu
        self.recent_files_menu = file_menu.addMenu("Open Recent")
        for action in self.recent_file_actions:
            self.recent_files_menu.addAction(action)
        self.recent_files_menu.addSeparator()
        self.act_clear_recent = QAction("Clear Recent Files", self)
        self.act_clear_recent.triggered.connect(self._clear_recent_files)
        self.recent_files_menu.addAction(self.act_clear_recent)

        file_menu.addSeparator()
        file_menu.addAction(self.act_add_files)
        file_menu.addAction(self.act_add_folder)
        file_menu.addAction(self.act_new_folder)
        file_menu.addSeparator()
        file_menu.addAction(self.act_merge_file)
        file_menu.addSeparator()
        file_menu.addAction(self.act_file_properties)
        file_menu.addAction(self.act_repack)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        # Update recent files menu after creation
        self._update_recent_files_menu()

        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.act_expand)
        view_menu.addAction(self.act_collapse)
        view_menu.addSeparator()
        view_menu.addAction(self.act_plot_selected)
        view_menu.addSeparator()
        view_menu.addAction(self.act_increase_font)
        view_menu.addAction(self.act_decrease_font)
        view_menu.addAction(self.act_reset_font)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.act_about)

    # Determine where to add new content in the HDF5 file
    def _get_target_group_path(self) -> str:
        """Determine the target HDF5 group path for adding new content.

        Returns:
            String path to target group, or "/" for root
        """
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
        """Determine the target HDF5 group path for a given tree index.

        Args:
            index: QModelIndex of the tree item

        Returns:
            String path to target group, or "/" for root
        """
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
        """Open a file selection dialog and add selected files to the HDF5 archive."""
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

        # Update file size display after adding files
        self._update_file_size_display()

    def add_folder_dialog(self) -> None:
        """Open a folder selection dialog and add folder contents recursively to HDF5."""
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

        # Update file size display after adding folder
        self._update_file_size_display()

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

            # Update file size display after creating folder
            self._update_file_size_display()

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
        """Add multiple files and folders to the HDF5 archive in batch.

        Args:
            files: List of file paths to add
            folders: List of folder paths to add recursively
            target_group: HDF5 group path where items should be added

        Returns:
            Tuple of (added_count, error_list) where added_count is number of successfully added items
            and error_list contains error messages for failed items
        """
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

        Args:
            grp: HDF5 group or file object where the dataset should be created
            h5_path: Full HDF5 path for the new dataset
            disk_path: Path to the file on disk
            np: numpy module reference for array creation

        Raises:
            FileExistsError: If the path already exists in the HDF5 file
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

        Args:
            f: Open HDF5 file object
            h5_path: Desired HDF5 path for the CSV group (will have .csv extension removed)
            disk_path: Path to the CSV file on disk
        """
        # Create progress dialog
        progress = self._create_progress_dialog("Reading CSV file...")

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
            # Determine optimal chunk size for compression
            data_len = len(col_data)
            chunk_size = min(10000, data_len) if data_len > 1000 else None

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
                        compression_opts=6,  # Higher compression for text (1-9, 6 is good balance)
                        chunks=(chunk_size,) if chunk_size else True  # Enable chunking for compression
                    )
                except Exception:  # noqa: BLE001
                    # Fallback: convert to bytes with compression
                    str_list = [str(x) for x in col_data.values]
                    grp.create_dataset(
                        ds_name,
                        data=str_list,
                        dtype=h5py.string_dtype(encoding="utf-8"),
                        compression="gzip",
                        compression_opts=6,
                        chunks=(chunk_size,) if chunk_size else True
                    )
            else:
                # Numeric or other numpy-supported dtypes with compression
                # Use chunking to enable compression and improve I/O for partial reads
                grp.create_dataset(
                    ds_name,
                    data=col_data.values,
                    compression="gzip",
                    compression_opts=4,  # Moderate compression for numeric data (1-9)
                    chunks=(chunk_size,) if chunk_size else True
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
            self._update_file_size_display()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Failed to create file",
                f"Could not create HDF5 file:\n{filepath}\n\n{exc}",
            )

    def open_file_dialog(self) -> None:
        """Open a file selection dialog to open an existing HDF5 file."""
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
        """Load an HDF5 file for viewing and editing.

        Args:
            path: Path to the HDF5 file to load
        """
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

        # Update file size display
        self._update_file_size_display()

        # Add to recent files
        self._add_recent_file(path)

    def _update_file_size_display(self) -> None:
        """Update the file size display in the status bar."""
        if not self.model or not self.model.filepath:
            self.file_size_label.setText("")
            return

        try:
            file_path = Path(self.model.filepath)
            if file_path.exists():
                size_bytes = file_path.stat().st_size
                size_str = self._format_file_size(size_bytes)
                self.file_size_label.setText(f"File size: {size_str}")
            else:
                self.file_size_label.setText("")
        except Exception:
            self.file_size_label.setText("")

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.

        Args:
            size_bytes: File size in bytes

        Returns:
            Formatted string like '1.5 MB' or '234 KB'
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                if unit == 'B':
                    return f"{size_bytes:.0f} {unit}"
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def _get_recent_files(self) -> list[str]:
        """Get the list of recent files from QSettings.

        Returns:
            List of recent file paths, most recent first
        """
        recent_files = self.settings.value("recent_files", [])
        if not isinstance(recent_files, list):
            recent_files = []
        # Filter out files that no longer exist
        return [f for f in recent_files if os.path.exists(f)]

    def _add_recent_file(self, filepath: str) -> None:
        """Add a file to the recent files list.

        Args:
            filepath: Absolute path to the file to add
        """
        filepath = os.path.abspath(filepath)
        recent_files = self._get_recent_files()

        # Remove if already in list
        if filepath in recent_files:
            recent_files.remove(filepath)

        # Add to front of list
        recent_files.insert(0, filepath)

        # Limit to max_recent_files
        recent_files = recent_files[:self.max_recent_files]

        # Save to settings
        self.settings.setValue("recent_files", recent_files)

        # Update menu
        self._update_recent_files_menu()

    def _update_recent_files_menu(self) -> None:
        """Update the recent files menu with current list."""
        recent_files = self._get_recent_files()

        for i, action in enumerate(self.recent_file_actions):
            if i < len(recent_files):
                filepath = recent_files[i]
                # Show just the filename and first part of path for readability
                display_name = f"{i + 1}. {os.path.basename(filepath)}"
                action.setText(display_name)
                action.setData(filepath)
                action.setVisible(True)
                action.setToolTip(filepath)
            else:
                action.setVisible(False)

        # Enable/disable clear action and menu
        has_files = len(recent_files) > 0
        self.act_clear_recent.setEnabled(has_files)
        self.recent_files_menu.setEnabled(has_files)

    def _open_recent_file(self) -> None:
        """Open a file from the recent files list."""
        action = self.sender()
        if action:
            filepath = action.data()
            if filepath and os.path.exists(filepath):
                self.load_hdf5(filepath)
            else:
                # File no longer exists, remove from list
                recent_files = self._get_recent_files()
                if filepath in recent_files:
                    recent_files.remove(filepath)
                    self.settings.setValue("recent_files", recent_files)
                    self._update_recent_files_menu()
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"The file no longer exists:\n{filepath}"
                )

    def _clear_recent_files(self) -> None:
        """Clear the recent files list."""
        reply = QMessageBox.question(
            self,
            "Clear Recent Files",
            "Are you sure you want to clear the recent files list?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.settings.setValue("recent_files", [])
            self._update_recent_files_menu()
            self.statusBar().showMessage("Recent files list cleared", 3000)

    def show_about_dialog(self) -> None:
        """Show the About VibeHDF5 dialog."""
        try:
            from vibehdf5 import __version__
            version = __version__
        except ImportError:
            version = "unknown"

        about_text = f"""<h2>VibeHDF5</h2>
        <p><b>Version:</b> {version}</p>
        <p>A powerful, lightweight GUI application for browsing, managing, and visualizing HDF5 file structures.</p>

        <p><b>Features:</b></p>
        <ul>
        <li>Browse and explore HDF5 file hierarchies</li>
        <li>Preview datasets with syntax highlighting</li>
        <li>Import and manage CSV data with filtering and plotting</li>
        <li>Drag-and-drop file import and export</li>
        <li>Interactive matplotlib plotting</li>
        </ul>

        <p><b>Built with:</b></p>
        <ul>
        <li>PySide6 / PyQt6 (Qt for Python)</li>
        <li>h5py (HDF5 for Python)</li>
        <li>NumPy, Pandas, Matplotlib</li>
        </ul>

        <p><b>Author:</b> Jacob Williams</p>
        <p><b>Repository:</b> <a href="https://github.com/jacobwilliams/vibehdf5">github.com/jacobwilliams/vibehdf5</a></p>
        <p><b>License:</b> MIT</p>
        """

        QMessageBox.about(self, "About VibeHDF5", about_text)

    def _increase_font_size(self) -> None:
        """Increase the application font size."""
        app = QApplication.instance()
        if app:
            font = app.font()
            current_size = font.pointSize()
            if current_size > 0:  # pointSize returns -1 if not set
                new_size = min(current_size + 1, 32)  # Cap at 32pt
                font.setPointSize(new_size)
            else:
                # Use pixel size as fallback
                pixel_size = font.pixelSize()
                if pixel_size > 0:
                    new_size = min(pixel_size + 1, 42)  # Cap at 42px
                    font.setPixelSize(new_size)
                else:
                    # Default starting point
                    font.setPointSize(12)
            app.setFont(font)

            # Also update the preview_edit fixed-width font
            preview_font = self.preview_edit.font()
            if current_size > 0:
                preview_font.setPointSize(new_size)
            else:
                preview_font.setPixelSize(new_size)
            self.preview_edit.setFont(preview_font)

            self.statusBar().showMessage(f"Font size increased to {font.pointSize()}pt", 2000)

    def _decrease_font_size(self) -> None:
        """Decrease the application font size."""
        app = QApplication.instance()
        if app:
            font = app.font()
            current_size = font.pointSize()
            if current_size > 0:
                new_size = max(current_size - 1, 6)  # Minimum 6pt
                font.setPointSize(new_size)
            else:
                # Use pixel size as fallback
                pixel_size = font.pixelSize()
                if pixel_size > 0:
                    new_size = max(pixel_size - 1, 8)  # Minimum 8px
                    font.setPixelSize(new_size)
                else:
                    # Default starting point
                    font.setPointSize(10)
            app.setFont(font)

            # Also update the preview_edit fixed-width font
            preview_font = self.preview_edit.font()
            if current_size > 0:
                preview_font.setPointSize(new_size)
            else:
                preview_font.setPixelSize(new_size)
            self.preview_edit.setFont(preview_font)

            self.statusBar().showMessage(f"Font size decreased to {font.pointSize()}pt", 2000)

    def _reset_font_size(self) -> None:
        """Reset the application font size to default."""
        app = QApplication.instance()
        if app:
            # Get system default font
            default_font = QApplication.font()
            app.setFont(default_font)

            # Also reset the preview_edit fixed-width font
            try:
                fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            except Exception:
                fixed = QFont("Courier New")
            self.preview_edit.setFont(fixed)

            self.statusBar().showMessage(f"Font size reset to default ({default_font.pointSize()}pt)", 2000)

    def _repack_file_dialog(self) -> None:
        """Dialog to repack the HDF5 file to reclaim space from deleted items."""
        if not self.model or not self.model.filepath:
            QMessageBox.information(self, "No File", "No HDF5 file is currently loaded.")
            return

        current_path = Path(self.model.filepath)
        if not current_path.exists():
            QMessageBox.warning(self, "File Not Found", "The current file no longer exists.")
            return

        # Get current file size
        current_size = current_path.stat().st_size
        current_size_str = self._format_file_size(current_size)

        # Show confirmation dialog with explanation
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Repack HDF5 File")
        msg.setText(
            "Repack the HDF5 file to reclaim space from deleted items?\n\n"
            f"Current file size: {current_size_str}\n\n"
            "Note: When you delete datasets or groups, HDF5 marks the space as unused "
            "but doesn't immediately reclaim it. Repacking creates a new optimized file "
            "that reclaims this space.\n\n"
            "This operation may take a few moments for large files."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)

        if msg.exec() != QMessageBox.Yes:
            return

        # Perform the repack
        try:
            import tempfile
            import shutil

            # Create a temporary file for the repacked version
            temp_fd, temp_path = tempfile.mkstemp(suffix='.h5', prefix='repack_')
            os.close(temp_fd)  # Close the file descriptor, we'll let h5py handle it

            # Show progress dialog
            progress = QProgressDialog(
                "Repacking file...\nThis may take a moment for large files.",
                None,
                0,
                0,
                self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QApplication.processEvents()

            try:
                # Copy all data to a new file (this automatically repacks)
                with h5py.File(str(current_path), 'r') as source:
                    with h5py.File(temp_path, 'w') as dest:
                        # Copy all groups and datasets recursively
                        def copy_recursively(src_group, dst_group):
                            for key in src_group.keys():
                                if isinstance(src_group[key], h5py.Group):
                                    # Create group and copy attributes
                                    new_group = dst_group.create_group(key)
                                    for attr_key, attr_val in src_group[key].attrs.items():
                                        new_group.attrs[attr_key] = attr_val
                                    # Recurse into subgroup
                                    copy_recursively(src_group[key], new_group)
                                elif isinstance(src_group[key], h5py.Dataset):
                                    # Copy dataset with compression if available
                                    src_ds = src_group[key]
                                    dst_group.create_dataset(
                                        key,
                                        data=src_ds[()],
                                        dtype=src_ds.dtype,
                                        compression=src_ds.compression,
                                        compression_opts=src_ds.compression_opts
                                    )
                                    # Copy attributes
                                    for attr_key, attr_val in src_ds.attrs.items():
                                        dst_group[key].attrs[attr_key] = attr_val

                        # Copy root attributes
                        for attr_key, attr_val in source.attrs.items():
                            dest.attrs[attr_key] = attr_val

                        # Copy all contents
                        copy_recursively(source, dest)

                progress.close()

                # Get new file size
                new_size = Path(temp_path).stat().st_size
                new_size_str = self._format_file_size(new_size)
                saved_bytes = current_size - new_size
                saved_str = self._format_file_size(saved_bytes) if saved_bytes > 0 else "0 B"

                # Replace original file with repacked version
                shutil.move(temp_path, str(current_path))

                # Reload the file
                self.load_hdf5(str(current_path))

                # Show results
                result_msg = (
                    f"File successfully repacked!\n\n"
                    f"Original size: {current_size_str}\n"
                    f"New size: {new_size_str}\n"
                    f"Space reclaimed: {saved_str}"
                )

                if saved_bytes <= 0:
                    result_msg += "\n\nNo space was reclaimed. The file was already optimally packed."

                QMessageBox.information(self, "Repack Complete", result_msg)

            finally:
                # Clean up temp file if it still exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Repack Failed",
                f"Failed to repack the file:\n\n{exc}\n\nThe original file has not been modified."
            )

    def _show_file_properties_dialog(self) -> None:
        """Show a dialog with detailed file properties and metadata."""
        if not self.model or not self.model.filepath:
            QMessageBox.information(self, "No File", "No HDF5 file is currently loaded.")
            return

        file_path = Path(self.model.filepath)
        if not file_path.exists():
            QMessageBox.warning(self, "File Not Found", "The current file no longer exists.")
            return

        try:
            # Gather file information
            info = {}

            # Basic file info
            info['File Path'] = str(file_path.absolute())
            info['File Name'] = file_path.name

            # File size
            file_size = file_path.stat().st_size
            info['File Size'] = f"{self._format_file_size(file_size)} ({file_size:,} bytes)"

            # File timestamps
            import datetime
            mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
            info['Modified'] = mtime.strftime('%Y-%m-%d %H:%M:%S')
            ctime = datetime.datetime.fromtimestamp(file_path.stat().st_ctime)
            info['Created'] = ctime.strftime('%Y-%m-%d %H:%M:%S')

            # HDF5-specific information
            with h5py.File(str(file_path), 'r') as h5:
                # Count items
                num_groups = 0
                num_datasets = 0
                num_attrs = len(h5.attrs)
                total_datasets_size = 0

                def count_items(group):
                    nonlocal num_groups, num_datasets, num_attrs, total_datasets_size
                    for key in group.keys():
                        if isinstance(group[key], h5py.Group):
                            num_groups += 1
                            num_attrs += len(group[key].attrs)
                            count_items(group[key])
                        elif isinstance(group[key], h5py.Dataset):
                            num_datasets += 1
                            num_attrs += len(group[key].attrs)
                            try:
                                # Estimate dataset size
                                ds = group[key]
                                if hasattr(ds, 'nbytes'):
                                    total_datasets_size += ds.nbytes
                            except Exception:
                                pass

                count_items(h5)

                info['Groups'] = f"{num_groups:,}"
                info['Datasets'] = f"{num_datasets:,}"
                info['Attributes'] = f"{num_attrs:,}"
                info['Total Items'] = f"{num_groups + num_datasets:,}"

                # Dataset storage info
                if total_datasets_size > 0:
                    info['Dataset Size'] = f"{self._format_file_size(total_datasets_size)} ({total_datasets_size:,} bytes)"
                    overhead = file_size - total_datasets_size
                    if overhead > 0:
                        overhead_pct = (overhead / file_size) * 100
                        info['Metadata Overhead'] = f"{self._format_file_size(overhead)} ({overhead_pct:.1f}%)"

                # HDF5 library version
                try:
                    info['HDF5 Library'] = h5py.version.hdf5_version
                except Exception:
                    pass

                # h5py version
                try:
                    info['h5py Version'] = h5py.version.version
                except Exception:
                    pass

                # File format version
                try:
                    # Try to get userblock size
                    fid = h5.id
                    fcpl = fid.get_create_plist()
                    userblock_size = fcpl.get_userblock()
                    if userblock_size > 0:
                        info['Userblock Size'] = f"{self._format_file_size(userblock_size)}"
                except Exception:
                    pass

                # Root attributes
                if len(h5.attrs) > 0:
                    root_attrs = []
                    for key in list(h5.attrs.keys())[:5]:  # Show first 5
                        root_attrs.append(key)
                    info['Root Attributes'] = ', '.join(root_attrs)
                    if len(h5.attrs) > 5:
                        info['Root Attributes'] += f" (+{len(h5.attrs) - 5} more)"

            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("File Properties")
            dialog.setMinimumWidth(500)

            layout = QVBoxLayout(dialog)

            # Add header label
            header = QLabel(f"<b>HDF5 File Information</b>")
            header.setStyleSheet("font-size: 14px; padding: 5px;")
            layout.addWidget(header)

            # Create table for properties
            table = QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Property", "Value"])
            table.horizontalHeader().setStretchLastSection(True)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.verticalHeader().setVisible(False)

            # Populate table
            table.setRowCount(len(info))
            for i, (key, value) in enumerate(info.items()):
                # Property name
                key_item = QTableWidgetItem(key)
                key_item.setFont(QFont(key_item.font().family(), -1, QFont.Bold))
                table.setItem(i, 0, key_item)

                # Property value
                value_item = QTableWidgetItem(str(value))
                table.setItem(i, 1, value_item)

            table.resizeColumnsToContents()
            table.setColumnWidth(0, 180)  # Fixed width for property names

            layout.addWidget(table)

            # Add buttons
            button_box = QHBoxLayout()
            button_box.addStretch()

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            close_btn.setDefault(True)
            button_box.addWidget(close_btn)

            layout.addLayout(button_box)

            dialog.exec()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to retrieve file properties:\n\n{exc}"
            )

    def _show_dataset_info_dialog(self, dataset_path: str) -> None:
        """Show a dialog with detailed dataset information.

        Args:
            dataset_path: HDF5 path to the dataset
        """
        if not self.model or not self.model.filepath:
            QMessageBox.information(self, "No File", "No HDF5 file is currently loaded.")
            return

        try:
            with h5py.File(self.model.filepath, 'r') as h5:
                if dataset_path not in h5:
                    QMessageBox.warning(self, "Not Found", f"Dataset '{dataset_path}' not found.")
                    return

                ds = h5[dataset_path]
                if not isinstance(ds, h5py.Dataset):
                    QMessageBox.warning(self, "Not a Dataset", f"'{dataset_path}' is not a dataset.")
                    return

                # Gather dataset information
                info = {}

                # Basic info
                info['Name'] = ds.name
                info['Shape'] = str(ds.shape)
                info['Data Type'] = str(ds.dtype)
                info['Size'] = f"{ds.size:,} elements"

                # Memory size
                if hasattr(ds, 'nbytes'):
                    info['Memory Size'] = f"{self._format_file_size(ds.nbytes)} ({ds.nbytes:,} bytes)"

                # Storage size (actual disk space used)
                try:
                    storage_size = ds.id.get_storage_size()
                    if storage_size > 0:
                        info['Storage Size'] = f"{self._format_file_size(storage_size)} ({storage_size:,} bytes)"
                        # Calculate compression ratio
                        if hasattr(ds, 'nbytes') and ds.nbytes > 0:
                            ratio = ds.nbytes / storage_size
                            info['Compression Ratio'] = f"{ratio:.2f}:1"
                except Exception:
                    pass

                # Chunks
                if ds.chunks:
                    info['Chunks'] = str(ds.chunks)
                    info['Chunked'] = 'Yes'
                else:
                    info['Chunked'] = 'No (Contiguous)'

                # Compression
                compression = ds.compression
                if compression:
                    info['Compression'] = compression
                    if ds.compression_opts:
                        info['Compression Options'] = str(ds.compression_opts)
                else:
                    info['Compression'] = 'None'

                # Filters
                try:
                    if ds.scaleoffset:
                        info['Scale-Offset Filter'] = str(ds.scaleoffset)
                except Exception:
                    pass

                try:
                    if ds.shuffle:
                        info['Shuffle Filter'] = 'Enabled'
                except Exception:
                    pass

                try:
                    if ds.fletcher32:
                        info['Fletcher32 Checksum'] = 'Enabled'
                except Exception:
                    pass

                # Fill value
                try:
                    fillvalue = ds.fillvalue
                    if fillvalue is not None:
                        info['Fill Value'] = str(fillvalue)
                except Exception:
                    pass

                # Attributes
                num_attrs = len(ds.attrs)
                info['Attributes'] = f"{num_attrs:,}"
                if num_attrs > 0:
                    attr_names = []
                    for key in list(ds.attrs.keys())[:5]:  # Show first 5
                        attr_names.append(key)
                    info['Attribute Names'] = ', '.join(attr_names)
                    if num_attrs > 5:
                        info['Attribute Names'] += f" (+{num_attrs - 5} more)"

                # External storage
                try:
                    external = ds.external
                    if external:
                        info['External Storage'] = f"{len(external)} file(s)"
                except Exception:
                    pass

                # Dimensions (for multidimensional datasets)
                if len(ds.shape) > 1:
                    info['Dimensions'] = f"{len(ds.shape)}D"
                    for i, dim_size in enumerate(ds.shape):
                        info[f'  Dimension {i}'] = f"{dim_size:,}"

                # For numeric data, show value range if dataset is small enough
                if ds.size > 0 and ds.size <= 1000000 and ds.dtype.kind in ('i', 'u', 'f'):
                    try:
                        data = ds[:]
                        if data.size > 0:
                            info['Min Value'] = str(np.min(data))
                            info['Max Value'] = str(np.max(data))
                            if ds.dtype.kind == 'f':
                                info['Mean Value'] = f"{np.mean(data):.6g}"
                                info['Std Dev'] = f"{np.std(data):.6g}"
                    except Exception:
                        pass

                # Create dialog
                dialog = QDialog(self)
                dialog.setWindowTitle(f"Dataset Information: {ds.name.split('/')[-1]}")
                dialog.setMinimumWidth(500)

                layout = QVBoxLayout(dialog)

                # Add header label
                header = QLabel(f"<b>Dataset: {dataset_path}</b>")
                header.setStyleSheet("font-size: 14px; padding: 5px;")
                layout.addWidget(header)

                # Create table for properties
                table = QTableWidget()
                table.setColumnCount(2)
                table.setHorizontalHeaderLabels(["Property", "Value"])
                table.horizontalHeader().setStretchLastSection(True)
                table.setAlternatingRowColors(True)
                table.setEditTriggers(QAbstractItemView.NoEditTriggers)
                table.setSelectionBehavior(QAbstractItemView.SelectRows)
                table.verticalHeader().setVisible(False)

                # Populate table
                table.setRowCount(len(info))
                for i, (key, value) in enumerate(info.items()):
                    # Property name
                    key_item = QTableWidgetItem(key)
                    if not key.startswith('  '):  # Don't bold sub-items
                        key_item.setFont(QFont(key_item.font().family(), -1, QFont.Bold))
                    table.setItem(i, 0, key_item)

                    # Property value
                    value_item = QTableWidgetItem(str(value))
                    table.setItem(i, 1, value_item)

                table.resizeColumnsToContents()
                table.setColumnWidth(0, 180)  # Fixed width for property names

                layout.addWidget(table)

                # Add buttons
                button_box = QHBoxLayout()
                button_box.addStretch()

                close_btn = QPushButton("Close")
                close_btn.clicked.connect(dialog.accept)
                close_btn.setDefault(True)
                button_box.addWidget(close_btn)

                layout.addLayout(button_box)

                dialog.exec()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to retrieve dataset information:\n\n{exc}"
            )

    def _merge_file_dialog(self) -> None:
        """Dialog to select and merge another HDF5 file into the current file."""
        if not self.model or not self.model.filepath:
            QMessageBox.information(self, "No File", "No HDF5 file is currently loaded. Open or create a file first.")
            return

        current_path = Path(self.model.filepath)
        if not current_path.exists():
            QMessageBox.warning(self, "File Not Found", "The current file no longer exists.")
            return

        # Select source file to merge from
        source_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File to Merge",
            str(current_path.parent),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )

        if not source_file:
            return  # User cancelled

        source_path = Path(source_file)

        # Don't allow merging file with itself
        if source_path.absolute() == current_path.absolute():
            QMessageBox.warning(self, "Invalid Selection", "Cannot merge a file with itself.")
            return

        # Get basic info about source file
        try:
            with h5py.File(str(source_path), 'r') as src_h5:
                num_groups = 0
                num_datasets = 0

                def count_items(group):
                    nonlocal num_groups, num_datasets
                    for key in group.keys():
                        if isinstance(group[key], h5py.Group):
                            num_groups += 1
                            count_items(group[key])
                        elif isinstance(group[key], h5py.Dataset):
                            num_datasets += 1

                count_items(src_h5)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Cannot Read File",
                f"Failed to read source file:\n\n{exc}"
            )
            return

        # Confirm merge operation
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Confirm Merge")
        msg.setText(
            f"Merge contents from:\n{source_path.name}\n\n"
            f"Into current file:\n{current_path.name}\n\n"
            f"Source contains: {num_groups} group(s), {num_datasets} dataset(s)\n\n"
            "Note: If items with the same path exist in both files, "
            "you will be prompted for each conflict."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)

        if msg.exec() != QMessageBox.Yes:
            return

        # Perform the merge
        try:
            conflicts = []
            items_copied = 0

            # Show progress dialog
            progress = QProgressDialog(
                "Merging files...\nThis may take a moment for large files.",
                "Cancel",
                0,
                num_groups + num_datasets,
                self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QApplication.processEvents()

            items_processed = 0

            with h5py.File(str(source_path), 'r') as src_h5:
                with h5py.File(str(current_path), 'r+') as dst_h5:

                    def merge_recursively(src_group, dst_group, path="/"):
                        nonlocal items_copied, items_processed, conflicts

                        if progress.wasCanceled():
                            return False

                        for key in src_group.keys():
                            if progress.wasCanceled():
                                return False

                            src_path = posixpath.join(path, key)

                            if isinstance(src_group[key], h5py.Group):
                                items_processed += 1
                                progress.setValue(items_processed)
                                progress.setLabelText(f"Merging: {src_path}")
                                QApplication.processEvents()

                                # Check if group exists
                                if key in dst_group:
                                    if isinstance(dst_group[key], h5py.Group):
                                        # Merge into existing group
                                        if not merge_recursively(src_group[key], dst_group[key], src_path):
                                            return False
                                    else:
                                        # Conflict: target is a dataset
                                        conflicts.append(f"{src_path} (group conflicts with existing dataset)")
                                else:
                                    # Create new group and copy attributes
                                    new_group = dst_group.create_group(key)
                                    for attr_key, attr_val in src_group[key].attrs.items():
                                        try:
                                            new_group.attrs[attr_key] = attr_val
                                        except Exception:
                                            pass
                                    items_copied += 1
                                    # Recurse into subgroup
                                    if not merge_recursively(src_group[key], new_group, src_path):
                                        return False

                            elif isinstance(src_group[key], h5py.Dataset):
                                items_processed += 1
                                progress.setValue(items_processed)
                                progress.setLabelText(f"Merging: {src_path}")
                                QApplication.processEvents()

                                # Check if dataset exists
                                if key in dst_group:
                                    # Ask user what to do
                                    reply = QMessageBox.question(
                                        self,
                                        "Item Exists",
                                        f"Item already exists: {src_path}\n\nOverwrite?",
                                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                        QMessageBox.No
                                    )

                                    if reply == QMessageBox.Cancel:
                                        return False
                                    elif reply == QMessageBox.Yes:
                                        # Delete existing and copy new
                                        del dst_group[key]
                                        src_ds = src_group[key]
                                        dst_group.create_dataset(
                                            key,
                                            data=src_ds[()],
                                            dtype=src_ds.dtype,
                                            compression=src_ds.compression,
                                            compression_opts=src_ds.compression_opts
                                        )
                                        # Copy attributes
                                        for attr_key, attr_val in src_ds.attrs.items():
                                            try:
                                                dst_group[key].attrs[attr_key] = attr_val
                                            except Exception:
                                                pass
                                        items_copied += 1
                                    else:
                                        # Skip this item
                                        conflicts.append(f"{src_path} (skipped - already exists)")
                                else:
                                    # Copy dataset
                                    src_ds = src_group[key]
                                    dst_group.create_dataset(
                                        key,
                                        data=src_ds[()],
                                        dtype=src_ds.dtype,
                                        compression=src_ds.compression,
                                        compression_opts=src_ds.compression_opts
                                    )
                                    # Copy attributes
                                    for attr_key, attr_val in src_ds.attrs.items():
                                        try:
                                            dst_group[key].attrs[attr_key] = attr_val
                                        except Exception:
                                            pass
                                    items_copied += 1

                        return True

                    # Copy root attributes if they don't exist
                    for attr_key, attr_val in src_h5.attrs.items():
                        if attr_key not in dst_h5.attrs:
                            try:
                                dst_h5.attrs[attr_key] = attr_val
                            except Exception:
                                pass

                    # Merge contents
                    success = merge_recursively(src_h5, dst_h5)

            progress.close()

            if not success:
                QMessageBox.information(self, "Merge Cancelled", "Merge operation was cancelled.")
                # Still reload to show any partial changes
                self.model.load_file(str(current_path))
                self.tree.expandToDepth(2)
                self._update_file_size_display()
                return

            # Reload the file
            self.model.load_file(str(current_path))
            self.tree.expandToDepth(2)
            self._update_file_size_display()

            # Show results
            result_msg = f"Merge complete!\n\nItems copied: {items_copied}"

            if conflicts:
                result_msg += f"\n\nConflicts/Skipped: {len(conflicts)}"
                if len(conflicts) <= 10:
                    result_msg += "\n" + "\n".join(conflicts)
                else:
                    result_msg += "\n" + "\n".join(conflicts[:10]) + f"\n... and {len(conflicts) - 10} more"

            QMessageBox.information(self, "Merge Complete", result_msg)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Merge Failed",
                f"Failed to merge files:\n\n{exc}"
            )

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
        """Handle tree selection changes and update preview.

        Args:
            selected: QItemSelection of newly selected items
            _deselected: QItemSelection of deselected items (unused)
        """
        indexes = selected.indexes()
        if not indexes:
            self._hide_attributes()
            # Clear plot display when nothing is selected
            self._current_csv_group_path = None
            self._saved_plots = []
            self._refresh_saved_plots_list()
            self._clear_plot_display()
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
            # Clear plot display for unknown item types
            self._current_csv_group_path = None
            self._saved_plots = []
            self._refresh_saved_plots_list()
            self._clear_plot_display()

    def _on_tree_item_renamed(self, topLeft, bottomRight, roles):
        """Handle when a tree item is renamed.

        Update internal references if the currently viewed CSV group was renamed.

        Args:
            topLeft: Top-left index of changed data
            bottomRight: Bottom-right index of changed data
            roles: List of roles that changed
        """
        # Check if this is an edit role change (rename)
        if Qt.EditRole not in roles and Qt.DisplayRole not in roles:
            return

        # Get the renamed item
        item = self.model.itemFromIndex(topLeft)
        if item is None:
            return

        new_path = item.data(self.model.ROLE_PATH)

        # If the currently viewed CSV group was renamed, update the reference
        if self._current_csv_group_path and new_path:
            # Check if we need to update the current CSV group path
            # This could be the group itself or a parent group
            old_path = self._current_csv_group_path

            # Try to determine the old path by checking against the new path
            # The model's setData already updated paths, so we need to check
            # if this rename affects our currently viewed path
            # For now, just clear it to be safe - the user can reselect
            # A more sophisticated approach would track the old path before rename

            # Actually, let's just refresh the selection to update paths correctly
            current_index = self.tree.currentIndex()
            if current_index.isValid():
                self.on_selection_changed(
                    self.tree.selectionModel().selection(),
                    self.tree.selectionModel().selection()
                )

    # Context menu handling
    def on_tree_context_menu(self, point) -> None:
        """Handle context menu requests on tree items.

        Args:
            point: QPoint position where the context menu was requested
        """
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

        # Add dataset information option
        act_info = None
        if kind == "dataset":
            act_info = menu.addAction(f"Dataset Information...")
            menu.addSeparator()

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
        if not act_info and not act_toggle_csv and not act_delete:
            return

        global_pos = self.tree.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)

        if chosen and chosen == act_info and act_info is not None:
            self._show_dataset_info_dialog(path)
        elif chosen == act_toggle_csv:
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
        """Delete an HDF5 item (dataset, group, or attribute).

        Args:
            kind: Type of item to delete ('dataset', 'group', or 'attr')
            path: HDF5 path to the item or its owner (for attributes)
            attr_key: Attribute key name if kind is 'attr', None otherwise
        """
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

            # Update file size display after deletion
            self._update_file_size_display()
        except Exception as exc:
            QMessageBox.warning(
                self, "Refresh failed", f"Deleted, but failed to refresh view: {exc}"
            )

    def preview_dataset(self, dspath: str) -> None:
        """Preview an HDF5 dataset in the preview pane.

        Args:
            dspath: HDF5 path to the dataset
        """
        # Clear plot display and saved plots when viewing a dataset
        self._current_csv_group_path = None
        self._saved_plots = []
        self._refresh_saved_plots_list()
        self._clear_plot_display()

        self.preview_label.setText(f"Dataset: {os.path.basename(dspath)}")
        fpath = self.model.filepath
        if not fpath:
            self._set_preview_text("No file loaded")
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
            self._hide_attributes()
            return
        # If the dataset name is an image format, try to display as image
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp', '.ico')
        if dspath.lower().endswith(image_extensions):
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
                            self._set_preview_text("Dataset is not a valid image byte array.")
                            self.preview_edit.setVisible(True)
                            self.preview_image.setVisible(False)
                            self._hide_attributes()
                            return
                    elif isinstance(data, bytes):
                        img_bytes = data
                    elif hasattr(data, "tobytes"):
                        img_bytes = data.tobytes()
                    else:
                        self._set_preview_text("Dataset is not a valid image byte array.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        self._hide_attributes()
                        return
                    pixmap = QPixmap()
                    # QPixmap.loadFromData will auto-detect the format
                    if pixmap.loadFromData(img_bytes):
                        # Scale pixmap to fit preview area, maintaining aspect ratio
                        self._show_scaled_image(pixmap)
                        self.preview_image.setVisible(True)
                        self.preview_edit.setVisible(False)
                        self.preview_table.setVisible(False)
                        self.filter_panel.setVisible(False)
                        # Show attributes for the dataset
                        self._show_attributes(obj)
                    else:
                        self._set_preview_text("Failed to load image from dataset.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        self._hide_attributes()
            except Exception as exc:
                self._set_preview_text(f"Error reading image dataset:\n{exc}")
                self.preview_edit.setVisible(True)
                self.preview_image.setVisible(False)
                self._hide_attributes()
            return
        # Otherwise, show text preview for non-image datasets
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
                if note:
                    note = note.strip('()')  # Remove parentheses from note
                    # add the note after the file name:
                    self.preview_label.setText(f"Dataset: {os.path.basename(dspath)} ({note})")
                # Apply syntax highlighting based on file extension
                language = get_language_from_path(dspath)
                self._set_preview_text(text=text, language=language)
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
        """Display a scaled image in the preview pane.

        Args:
            pixmap: QPixmap to display, or None to use stored pixmap
        """
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

    def updateCanvas(self):
        """If plot canvas is visible, apply tight layout and redraw"""
        if hasattr(self, 'plot_canvas') and self.plot_canvas.isVisible():
            try:
                self.plot_figure.tight_layout()
                self.plot_canvas.draw()
            except Exception:
                pass  # Ignore layout errors during resize

    def resizeEvent(self, event):
        """Handle window resize events to rescale displayed images and adjust plot layout.

        Args:
            event: QResizeEvent object
        """
        # If an image is visible, rescale it to fit the new size
        if self.preview_image.isVisible():
            self._show_scaled_image()

        self.updateCanvas()

        super().resizeEvent(event)

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        """Handle splitter movement to adjust plot layout.

        Args:
            pos: New position of the splitter handle
            index: Index of the splitter handle that was moved
        """

        self.updateCanvas()

    def preview_attribute(self, grouppath: str, key: str) -> None:
        """Preview an HDF5 attribute value.

        Args:
            grouppath: HDF5 path to the group or dataset containing the attribute
            key: Attribute key name
        """
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
                    self._clear_plot_display()
                    self._set_preview_text("(No content to display)")
                    # Show attributes for the group
                    self._show_attributes(grp)
        except Exception as exc:
            self._set_preview_text(f"Error reading group:\n{exc}")
            self._hide_attributes()

    def _get_th_location(self, ds_key, grp):
        """Get the location of a dataset, checking for optional 'Time History' subgroup.

        Args:
            ds_key: Dataset key name to look for
            grp: HDF5 group to search in

        Returns:
            Tuple of (key_in_group: bool, th_grp: h5py.Group)
        """
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
            progress = self._create_progress_dialog("Loading CSV metadata...")

            # First pass: only get metadata (dataset paths and row counts) - don't load data yet
            dataset_info = {}  # Maps column name to (ds_key, th_grp_path, row_count, dtype)
            max_rows = 0
            for idx, col_name in enumerate(col_names):
                if progress.wasCanceled():
                    progress.close()
                    self._set_preview_text("(CSV display cancelled)")
                    return

                # Update progress
                progress_val = int((idx / total_cols) * 30)
                progress.setValue(progress_val)
                progress.setLabelText(f"Reading metadata {idx + 1}/{total_cols}: {col_name}")
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
                        # Only get shape and dtype, don't load data
                        if len(ds.shape) > 0:
                            row_count = ds.shape[0]
                        else:
                            row_count = 1
                        # Store group path as string instead of group object
                        th_grp_path = th_grp.name
                        dataset_info[col_name] = (ds_key, th_grp_path, row_count, ds.dtype)
                        max_rows = max(max_rows, row_count)

            if not dataset_info:
                progress.close()
                self._set_preview_text("(No datasets found in CSV group)")
                return

            # Store dataset info and metadata for lazy loading
            self._csv_dataset_info = dataset_info  # For lazy loading
            self._csv_data_dict = {}  # Will be populated on-demand

            self._csv_column_names = col_names
            self._csv_total_rows = max_rows
            self._csv_sort_specs = []  # Initialize sort specs

            # Setup table - disable updates for performance
            progress.setLabelText("Setting up table...")
            progress.setValue(40)
            QApplication.processEvents()

            self.preview_table.setUpdatesEnabled(False)
            self.preview_table.setSortingEnabled(False)
            # Clear existing table content efficiently
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            self.preview_table.clear()
            # Reset lazy loading state
            self._table_loaded_rows = 0
            # Now set up the new table
            self.preview_table.setRowCount(max_rows)  # Set full row count for scrollbar
            self.preview_table.setColumnCount(len(col_names))
            self.preview_table.setHorizontalHeaderLabels(col_names)

            # Load only the data needed for first batch of rows
            initial_batch = min(self._table_batch_size, max_rows)
            progress.setLabelText(f"Loading initial {initial_batch} rows...")
            progress.setValue(50)
            QApplication.processEvents()

            # Lazy load columns as needed for initial batch
            # Need to access file separately since grp is from outer context
            fpath = self.model.filepath
            if fpath:
                with h5py.File(fpath, "r") as h5:
                    self._lazy_load_columns(col_names, 0, initial_batch, h5)
            self._populate_table_rows(0, initial_batch, self._csv_data_dict, col_names)
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

            # Load saved filters from HDF5 group (or clear if none exist)
            saved_filters = self._load_filters_from_hdf5(grp)
            if saved_filters:
                self._csv_filters = saved_filters
                self.statusBar().showMessage(
                    f"Loaded {len(saved_filters)} saved filter(s) from HDF5 file", 5000
                )
            else:
                self._csv_filters = []

            # Load saved sort from HDF5 group (or clear if none exist)
            saved_sort = self._load_sort_from_hdf5(grp)
            if saved_sort:
                self._csv_sort_specs = saved_sort
                self.btn_clear_sort.setEnabled(True)
                self.statusBar().showMessage(
                    f"Loaded sort by {len(saved_sort)} column(s) from HDF5 file", 5000
                )
            else:
                self._csv_sort_specs = []
                self.btn_clear_sort.setEnabled(False)

            # Load saved column visibility from HDF5 group
            saved_visible_columns = self._load_visible_columns_from_hdf5(grp)
            if saved_visible_columns:
                self._csv_visible_columns = saved_visible_columns
                self.statusBar().showMessage(
                    f"Loaded column visibility ({len(saved_visible_columns)}/{len(col_names)} columns) from HDF5 file", 5000
                )
            else:
                # Default to all columns visible
                self._csv_visible_columns = col_names.copy()

            # Always apply column visibility to ensure correct state
            self._apply_column_visibility()

            # Update model with visible columns for drag-and-drop export
            if self._current_csv_group_path and self.model:
                self.model.set_csv_visible_columns(self._current_csv_group_path, self._csv_visible_columns)

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

            # Reset scrollbar to top when switching CSV datasets
            self.preview_table.verticalScrollBar().setValue(0)

        except Exception as exc:
            if progress:
                progress.close()
            self._set_preview_text(f"Error displaying CSV table:\n{exc}")
            self.preview_table.setVisible(False)
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
            self._hide_attributes()

    def _ensure_all_data_loaded(self) -> None:
        """Ensure all CSV data is loaded (used before filtering/sorting/plotting)."""
        if not self._csv_dataset_info or not self._current_csv_group_path:
            return

        if not self.model or not self.model.filepath:
            return

        # Check if we need to load more data
        total_rows = getattr(self, '_csv_total_rows', 0)
        if total_rows == 0:
            return

        # Check if all columns are fully loaded
        all_loaded = True
        for col_name in self._csv_column_names:
            if col_name not in self._csv_data_dict or len(self._csv_data_dict[col_name]) < total_rows:
                all_loaded = False
                break

        if all_loaded:
            return  # Already loaded

        # Load remaining data with progress dialog
        progress = self._create_progress_dialog("Loading all CSV data for operation...", min_duration=200)

        try:
            with h5py.File(self.model.filepath, "r") as h5:
                if self._current_csv_group_path in h5:
                    grp = h5[self._current_csv_group_path]
                    if isinstance(grp, h5py.Group):
                        # Load all columns completely
                        progress.setLabelText("Loading complete dataset...")
                        progress.setValue(10)
                        QApplication.processEvents()

                        self._lazy_load_columns(self._csv_column_names, 0, total_rows, h5)

                        progress.setValue(100)
        except Exception as exc:
            self.statusBar().showMessage(f"Error loading data: {exc}", 5000)
        finally:
            progress.close()

    def _lazy_load_columns(self, col_names: list[str], start_row: int, end_row: int, h5: h5py.File) -> None:
        """Lazy load only the columns and rows needed from HDF5.

        Args:
            col_names: List of column names to load
            start_row: Starting row index
            end_row: Ending row index (exclusive)
            h5: Open HDF5 file object
        """
        for col_name in col_names:
            # Check if we need to load or extend this column
            if col_name in self._csv_data_dict:
                existing_data = self._csv_data_dict[col_name]
                # If already fully loaded for this range, skip
                if len(existing_data) >= end_row:
                    continue
                # Otherwise we need to extend - adjust start_row
                actual_start = len(existing_data)
            else:
                actual_start = 0  # Load from beginning

            # Load this column's data for the requested range
            if col_name in self._csv_dataset_info:
                ds_key, th_grp_path, row_count, dtype = self._csv_dataset_info[col_name]
                # Get the group from the file using the stored path
                th_grp = h5[th_grp_path]
                ds = th_grp[ds_key]

                if isinstance(ds, h5py.Dataset):
                    # Determine the actual slice to load
                    load_start = max(actual_start, start_row)
                    load_end = min(end_row, row_count)

                    if load_start >= load_end:
                        continue  # Nothing to load

                    # Load only the slice we need
                    if len(ds.shape) > 0:
                        data = ds[load_start:load_end]
                    else:
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

                        # Store or concatenate the data
                        if col_name in self._csv_data_dict:
                            existing_data = self._csv_data_dict[col_name]
                            # If there's a gap between existing and new data, pad with empty strings
                            if load_start > len(existing_data):
                                gap = np.array([""] * (load_start - len(existing_data)), dtype=object)
                                self._csv_data_dict[col_name] = np.concatenate([existing_data, gap, data])
                            else:
                                self._csv_data_dict[col_name] = np.concatenate([existing_data, data])
                        else:
                            # First load for this column
                            if load_start > 0:
                                # Need to pad the beginning if not starting from row 0
                                padding = np.array([""] * load_start, dtype=object)
                                self._csv_data_dict[col_name] = np.concatenate([padding, data])
                            else:
                                self._csv_data_dict[col_name] = data
                    else:
                        # Scalar dataset
                        if isinstance(data, bytes):
                            data = data.decode("utf-8", errors="replace")
                        self._csv_data_dict[col_name] = np.array([data], dtype=object)

    def _populate_table_rows(self, start_row: int, end_row: int, data_dict: dict, col_names: list[str]) -> None:
        """Populate table rows from start_row to end_row (exclusive).

        Args:
            start_row: Starting row index (inclusive)
            end_row: Ending row index (exclusive)
            data_dict: Dictionary mapping column names to data arrays
            col_names: List of column names in display order
        """
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
        """Handle table scroll events to load more rows as needed.

        Args:
            value: Scroll position value
        """
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
        """Load all rows from current position up to target_row.

        Args:
            target_row: Target row index to load up to
        """
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

            # Lazy load the data for this range if needed
            if self._current_csv_group_path and self.model and self.model.filepath:
                try:
                    with h5py.File(self.model.filepath, "r") as h5:
                        if self._current_csv_group_path in h5:
                            grp = h5[self._current_csv_group_path]
                            if isinstance(grp, h5py.Group):
                                self._lazy_load_columns(self._csv_column_names, start_row, end_row, h5)
                except Exception as exc:
                    self.statusBar().showMessage(f"Error loading data: {exc}", 5000)

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
        """Get the indices of selected columns in the CSV table.

        Returns:
            Sorted list of selected column indices (excluding hidden columns)
        """
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
            # Unique and sorted for stable behavior, excluding hidden columns
            return sorted({c for c in cols if c >= 0 and not self.preview_table.isColumnHidden(c)})
        except Exception:  # noqa: BLE001
            return []

    def _update_plot_action_enabled(self) -> None:
        """Update the enabled state of plot actions based on selection."""
        # Enable plotting when a CSV group is active and >= 1 column is selected
        is_csv = self._current_csv_group_path is not None and self.preview_table.isVisible()
        sel_cols = self._get_selected_column_indices() if is_csv else []
        self.act_plot_selected.setEnabled(is_csv and len(sel_cols) >= 1)

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

        - If 1 column selected: Y-axis is that column, X-axis is point count (0, 1, 2, ...)
        - If 2+ columns: First selected (or current) column is X, others are Y series
        - Adds legend and shows plot in embedded canvas
        """
        if self._current_csv_group_path is None or not self.preview_table.isVisible():
            QMessageBox.information(self, "Plot", "No CSV table is active to plot.")
            return

        # Ensure all data is loaded before plotting
        self._ensure_all_data_loaded()

        # Determine selected columns and their order
        sel_cols = self._get_selected_column_indices()
        if len(sel_cols) < 1:
            QMessageBox.information(
                self,
                "Plot",
                "Select at least one column to plot.",
            )
            return

        # Handle single column selection: use point count as x-axis
        if len(sel_cols) == 1:
            x_idx = None  # No x column - use point count
            y_idxs = sel_cols
        else:
            # Use current column as X if part of selection; else use the leftmost selected
            current_col = self.preview_table.currentColumn()
            x_idx = current_col if current_col in sel_cols else min(sel_cols)
            y_idxs = [c for c in sel_cols if c != x_idx]

        # Ensure y_idxs is a list (defensive check - do this BEFORE any operations on it)
        if isinstance(y_idxs, int):
            y_idxs = [y_idxs]

        # Column names from headers
        headers = [
            self.preview_table.horizontalHeaderItem(i).text()
            if self.preview_table.horizontalHeaderItem(i) is not None
            else f"col_{i}"
            for i in range(self.preview_table.columnCount())
        ]
        try:
            x_name = headers[x_idx] if x_idx is not None else "Point"
            y_names = [headers[i] for i in y_idxs]
        except Exception as e:
            QMessageBox.warning(self, "Plot", f"Failed to resolve column headers for plotting.\n\nError: {e}\ny_idxs type: {type(y_idxs)}, value: {y_idxs}")
            return

        # Use filtered data from the table instead of reading from HDF5
        # This ensures we only plot what's visible (respecting filters)
        if not self._csv_data_dict or self._csv_filtered_indices is None:
            QMessageBox.warning(self, "Plot", "No CSV data available.")
            return

        # Get the actual data length after loading
        actual_data_len = max(len(self._csv_data_dict[col]) for col in self._csv_data_dict) if self._csv_data_dict else 0

        # Filter indices to only valid range (in case of partial loading)
        valid_filtered_indices = self._csv_filtered_indices[self._csv_filtered_indices < actual_data_len]

        col_data = {}
        columns_to_load = y_names if x_idx is None else [x_name] + y_names
        for name in columns_to_load:
            if name in self._csv_data_dict:
                # Get only the filtered rows
                full_data = self._csv_data_dict[name]
                if isinstance(full_data, np.ndarray):
                    col_data[name] = full_data[valid_filtered_indices]
                else:
                    col_data[name] = np.array([full_data[i] for i in valid_filtered_indices if i < len(full_data)])

        if not any(name in col_data for name in y_names):
            QMessageBox.warning(self, "Plot", "Failed to get selected columns for plotting.")
            return

        # Prepare numeric data, align lengths
        try:
            # Determine minimum length from Y columns
            min_len = min(len(col_data.get(n, [])) for n in y_names if n in col_data)
            if min_len <= 0:
                QMessageBox.warning(self, "Plot", "No data to plot.")
                return

            # Process X-axis data using helper method (reuses logic from saved plots)
            plot_options = {}  # Quick plot uses default options
            x_arr, x_num, x_is_string, xaxis_datetime, min_len = self._process_x_axis_data(
                x_idx, col_data, y_names, x_name, plot_options
            )

            if min_len <= 0:
                QMessageBox.warning(self, "Plot", "No data to plot.")
                return

            # Clear the previous plot
            self.plot_figure.clear()

            # For immediate plotting (without saved config), use default light background
            ax = self.plot_figure.add_subplot(111)
            self._apply_plot_style(self.plot_figure, ax, use_dark=False)

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

            # Set axis labels and title
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
                title_obj = ax.set_title(title)
                title_obj.set_color('black')
            except Exception:
                pass
            legend = ax.legend()
            self._make_legend_interactive(ax, legend)
            ax.grid(True)

            # Format x-axis (datetime or categorical strings)
            self._format_xaxis(ax, self.plot_figure, xaxis_datetime, x_is_string, x_arr, min_len)

            self.plot_figure.tight_layout()

            # Refresh the canvas to display the plot
            self.plot_canvas.draw()

            # Switch to the Plot tab
            self.bottom_tabs.setCurrentIndex(1)

        except Exception as exc:  # noqa: BLE001
            import traceback
            tb_str = traceback.format_exc()
            QMessageBox.critical(self, "Plot error", f"Failed to plot data:\n{exc}\n\nTraceback:\n{tb_str}")

    def _get_hdf5_file_path(self) -> str | None:
        """Get the currently loaded HDF5 file path, or None if no file is loaded.

        Returns:
            str or None: File path if a file is loaded, None otherwise
        """
        return self.model.filepath if self.model else None

    def _configure_filters_dialog(self) -> None:
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

    def _show_statistics_dialog(self) -> None:
        """Open dialog to show statistics for CSV columns."""
        if not self._csv_column_names or (not self._csv_data_dict and not self._csv_dataset_info):
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        # Ensure all data is loaded before calculating statistics
        self._ensure_all_data_loaded()

        # Use filtered indices if available
        filtered_indices = self._csv_filtered_indices if hasattr(self, '_csv_filtered_indices') else None

        dialog = ColumnStatisticsDialog(
            self._csv_column_names,
            self._csv_data_dict,
            filtered_indices,
            self
        )
        dialog.exec()

    def _clear_filters(self) -> None:
        """Clear all active filters and show full dataset."""
        self._csv_filters = []
        self._save_filters_to_hdf5()
        self._apply_filters()

    def _configure_sort_dialog(self) -> None:
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

    def _clear_sort(self) -> None:
        """Clear all sorting and display data in original order."""
        self._csv_sort_specs = []
        self._save_sort_to_hdf5()
        self._apply_sort()

    def _save_sort_to_hdf5(self) -> None:
        """Save current sort specifications to the HDF5 file as a JSON attribute."""
        success_msg = f"Saved sort by {len(self._csv_sort_specs)} column(s) to HDF5 file" if self._csv_sort_specs else None
        clear_msg = "Cleared sort from HDF5 file"
        self._save_csv_attr_to_hdf5("csv_sort", self._csv_sort_specs, success_msg, clear_msg)

    def _load_sort_from_hdf5(self, grp: h5py.Group) -> list:
        """Load sort specifications from the HDF5 group attributes."""
        def validate_sort(specs):
            if isinstance(specs, list):
                return [tuple(spec) for spec in specs if isinstance(spec, list) and len(spec) == 2]
            return []
        return self._load_csv_attr_from_hdf5(grp, "csv_sort", validate_sort) or []

    def _configure_columns_dialog(self) -> None:
        """Open dialog to select which columns to display."""
        if not self._csv_column_names:
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        dialog = ColumnVisibilityDialog(
            self._csv_column_names,
            self._csv_visible_columns if self._csv_visible_columns else self._csv_column_names.copy(),
            self
        )

        if dialog.exec() == QDialog.Accepted:
            self._csv_visible_columns = dialog.get_visible_columns()
            self._save_visible_columns_to_hdf5()
            self._apply_column_visibility()

            # Update model so drag-and-drop exports only visible columns
            if self._current_csv_group_path and self.model:
                self.model.set_csv_visible_columns(self._current_csv_group_path, self._csv_visible_columns)

    def _apply_column_visibility(self) -> None:
        """Apply column visibility to the table."""
        if not self._csv_column_names:
            return

        # First, ensure all columns are visible (reset state)
        # This is important when switching between CSV groups
        for col_idx in range(self.preview_table.columnCount()):
            self.preview_table.setColumnHidden(col_idx, False)

        # Hide/show columns based on visibility list
        for col_idx, col_name in enumerate(self._csv_column_names):
            should_hide = col_name not in self._csv_visible_columns
            self.preview_table.setColumnHidden(col_idx, should_hide)

    def _on_column_header_context_menu(self, pos) -> None:
        """Handle right-click context menu on column header.

        Args:
            pos: QPoint position where the context menu was requested
        """
        # Get the column index at the clicked position
        header = self.preview_table.horizontalHeader()
        col_idx = header.logicalIndexAt(pos)

        if col_idx < 0 or col_idx >= len(self._csv_column_names):
            return

        col_name = self._csv_column_names[col_idx]

        # Create context menu
        menu = QMenu(self)
        act_unique = menu.addAction(f"Show Unique Values in '{col_name}'")

        # Show menu and handle selection
        global_pos = header.mapToGlobal(pos)
        chosen = menu.exec(global_pos)

        if chosen == act_unique:
            self._show_unique_values_dialog(col_name)

    def _show_unique_values_dialog(self, col_name: str) -> None:
        """Show dialog with unique values for a specific column.

        Args:
            col_name: Name of the column to show unique values for
        """
        # Ensure all data is loaded for accurate unique value calculation
        self._ensure_all_data_loaded()

        if not self._csv_data_dict or col_name not in self._csv_data_dict:
            QMessageBox.information(self, "No Data", f"Column '{col_name}' has no data loaded.")
            return

        # Get the column data
        col_data = self._csv_data_dict[col_name]

        # If filters are active, use only filtered rows
        if hasattr(self, '_csv_filtered_indices') and self._csv_filtered_indices is not None:
            # Ensure filtered indices are within bounds
            actual_data_len = len(col_data)
            valid_indices = self._csv_filtered_indices[self._csv_filtered_indices < actual_data_len]
            # Filter the data to only include visible rows
            filtered_data = col_data[valid_indices]
        else:
            filtered_data = col_data

        # Get unique values and sort them
        try:
            unique_values = np.unique(filtered_data)
            # Convert to list for display
            unique_list = [str(val) for val in unique_values]
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to get unique values: {e}")
            return

        # Show dialog
        dialog = UniqueValuesDialog(col_name, unique_list, self)
        dialog.exec()

    def _save_visible_columns_to_hdf5(self) -> None:
        """Save visible columns list to the HDF5 file as a JSON attribute."""
        # Only save if not all columns are visible
        value = self._csv_visible_columns if (self._csv_visible_columns and len(self._csv_visible_columns) < len(self._csv_column_names)) else None
        success_msg = f"Saved column visibility ({len(self._csv_visible_columns)}/{len(self._csv_column_names)} columns) to HDF5 file" if value else None
        clear_msg = "All columns visible - removed saved visibility preference"
        self._save_csv_attr_to_hdf5("csv_visible_columns", value, success_msg, clear_msg)

    def _load_visible_columns_from_hdf5(self, grp: h5py.Group):
        """Load visible columns list from the HDF5 group attributes.

        Args:
            grp: HDF5 group to load visibility settings from

        Returns:
            List of visible column names, or None if not saved
        """
        return self._load_csv_attr_from_hdf5(grp, "csv_visible_columns", lambda v: v if isinstance(v, list) else None)

    def _save_filters_to_hdf5(self) -> None:
        """Save current filters to the HDF5 file as a JSON attribute."""
        success_msg = f"Saved {len(self._csv_filters)} filter(s) to HDF5 file" if self._csv_filters else None
        clear_msg = "Cleared filters from HDF5 file"
        self._save_csv_attr_to_hdf5("csv_filters", self._csv_filters, success_msg, clear_msg)

    def _load_filters_from_hdf5(self, grp: h5py.Group) -> list:
        """Load filters from the HDF5 group attributes.

        Args:
            grp: HDF5 group to load filters from

        Returns:
            List of filters in format [column_name, operator, value]
        """
        def validate_filters(filters):
            if isinstance(filters, list):
                return [f for f in filters if isinstance(f, list) and len(f) == 3]
            return []
        return self._load_csv_attr_from_hdf5(grp, "csv_filters", validate_filters) or []

    def _apply_sort(self) -> None:
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

    def _apply_filters(self) -> None:
        """Apply current filters to the CSV table."""
        if not self._csv_data_dict and not self._csv_dataset_info:
            return

        # Ensure all data is loaded before filtering
        self._ensure_all_data_loaded()

        # Update filter status label
        if self._csv_filters:
            filter_text = f"{len(self._csv_filters)} filter(s) applied"
            self.filter_status_label.setText(filter_text)
            self.btn_clear_filters.setEnabled(True)
        else:
            self.filter_status_label.setText("No filters applied")
            self.btn_clear_filters.setEnabled(False)

        # Determine which rows pass all filters
        max_rows = max(len(self._csv_data_dict[col]) for col in self._csv_data_dict) if self._csv_data_dict else 0
        if max_rows == 0:
            return

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

    def _compare_values(self, left_value: any, operator: str, right_value: any) -> bool:
        """Compare two values using the specified operator.

        Handles numeric, datetime, and string comparisons.

        Args:
            left_value: Left side value (can be numeric, datetime, or string)
            operator: Comparison operator (==, !=, <, <=, >, >=)
            right_value: Right side value to compare against

        Returns:
            bool: Result of comparison, or True if comparison fails
        """
        # Mapping of operators to lambda functions for cleaner code
        ops = {
            "==": lambda a, b: a == b,
            "=": lambda a, b: a == b,  # Support both = and ==
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
        }

        if operator not in ops:
            return True

        try:
            # Try numeric comparison first
            left_num = float(left_value)
            right_num = float(right_value)
            return ops[operator](left_num, right_num)
        except (ValueError, TypeError):
            # Try datetime comparison
            try:
                left_dt = pd.to_datetime(str(left_value), errors="coerce")
                right_dt = pd.to_datetime(str(right_value), errors="coerce")
                if not pd.isna(left_dt) and not pd.isna(right_dt):
                    return ops[operator](left_dt, right_dt)
            except (ValueError, TypeError):
                pass

            # Fall back to string comparison
            try:
                return ops[operator](str(left_value), str(right_value))
            except Exception:
                return True

    def _evaluate_filter(self, col_data: np.ndarray | list, operator: str, value_str: str) -> np.ndarray:
        """Evaluate a filter condition on column data.

        Returns a boolean mask of the same length as col_data.
        """
        try:
            # Handle comparison operators using helper method
            if operator in ["==", "=", "!=", ">", ">=", "<", "<="]:
                # Use vectorized comparison via helper
                if isinstance(col_data, np.ndarray):
                    return np.array([self._compare_values(val, operator, value_str) for val in col_data])
                else:
                    return np.array([self._compare_values(val, operator, value_str) for val in col_data])

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

    def _save_plot_config_dialog(self) -> None:
        """Open dialog to save current plot configuration."""
        if not self._current_csv_group_path or not self.preview_table.isVisible():
            QMessageBox.information(
                self, "No CSV Data", "Load a CSV group and create a plot first."
            )
            return

        # Get currently selected columns
        sel_cols = self._get_selected_column_indices()
        if len(sel_cols) < 1:
            QMessageBox.information(
                self,
                "No Plot Selection",
                "Select at least one column before saving a plot configuration.",
            )
            return

        # Determine X and Y columns (same logic as plot_selected_columns)
        if len(sel_cols) == 1:
            x_idx = None  # Single column mode: use point count
            y_idxs = sel_cols
        else:
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

        # Capture current visibility state from plot if available
        series_visibility = self._capture_plot_visibility_state()

        plot_config = {
            "name": plot_name,
            "csv_group_path": self._current_csv_group_path,
            "column_names": column_names,
            "x_col_idx": x_idx,
            "y_col_idxs": y_idxs,
            "filtered_indices": filtered_indices,  # Store actual filtered row indices
            "start_row": start_row,  # Keep for backward compatibility
            "end_row": end_row,  # Keep for backward compatibility
            "csv_filters": self._csv_filters.copy() if self._csv_filters else [],  # Store filter specs
            "csv_sort": self._csv_sort_specs.copy() if hasattr(self, '_csv_sort_specs') and self._csv_sort_specs else [],  # Store sort specs
            "timestamp": time.time(),
            "plot_options": {
                "title": "",
                "xlabel": "",
                "ylabel": "",
                "grid": True,
                "legend": True,
                "series": {},  # Will be populated with per-series styles in the Edit Options dialog
                "series_visibility": series_visibility,  # Store current visibility state
            },
        }

        # Add to local list
        self._saved_plots.append(plot_config)

        # Save to HDF5
        self._save_plot_configs_to_hdf5()

        # Update list widget
        self._refresh_saved_plots_list()

        self.statusBar().showMessage(f"Saved plot configuration: {plot_name}", 3000)

    def _save_plot_configs_to_hdf5(self) -> None:
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

            # Update file size display after modification
            self._update_file_size_display()
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

        # Auto-select the first plot if available, or clear plot display if no plots
        if self._saved_plots:
            self.saved_plots_list.setCurrentRow(0)
        else:
            self._clear_plot_display()

    def _refresh_saved_plots_list(self) -> None:
        """Update the saved plots list widget with current configurations."""
        self.saved_plots_list.clear()

        # Get a standard chart/graph icon for plots
        style = QApplication.instance().style() if QApplication.instance() else None
        plot_icon = style.standardIcon(QStyle.SP_FileDialogContentsView) if style else None

        for plot_config in self._saved_plots:
            name = plot_config.get("name", "Unnamed Plot")
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)

            # Add plot icon to each item
            if plot_icon:
                item.setIcon(plot_icon)

            self.saved_plots_list.addItem(item)

        # Update button states
        self._update_plot_buttons_state()

    def _clear_plot_display(self) -> None:
        """Clear the plot display area."""
        self.plot_figure.clear()
        self.plot_canvas.draw()

    def _update_plot_buttons_state(self) -> None:
        """Enable/disable plot management buttons based on current state."""
        # Enable Save Plot button if CSV is loaded and columns are selected
        csv_loaded = self._current_csv_group_path is not None and self.preview_table.isVisible()
        sel_cols = self._get_selected_column_indices() if csv_loaded else []
        self.btn_save_plot.setEnabled(csv_loaded and len(sel_cols) >= 1)

        # Enable Delete and Edit Options buttons if a plot is selected
        has_selection = self.saved_plots_list.currentRow() >= 0
        self.btn_delete_plot.setEnabled(has_selection)
        self.btn_edit_plot_options.setEnabled(has_selection)

    def _on_saved_plot_selection_changed(self) -> None:
        """Handle selection change in saved plots list."""
        self._update_plot_buttons_state()

        # Automatically apply the selected plot
        current_item = self.saved_plots_list.currentItem()
        if current_item is not None:
            self._apply_saved_plot(current_item)

    def _on_saved_plot_clicked(self, item: QListWidgetItem) -> None:
        """Handle clicking on a saved plot item (even if already selected).

        Args:
            item: The list item that was clicked
        """
        # Apply the plot even if it's already selected
        if item is not None:
            self._apply_saved_plot(item)

    def _on_plot_item_renamed(self, item: QListWidgetItem):
        """Handle when a plot item is renamed by the user.

        Args:
            item: The list item that was edited
        """
        new_name = item.text().strip()
        if not new_name:
            # Don't allow empty names
            row = self.saved_plots_list.row(item)
            if 0 <= row < len(self._saved_plots):
                old_name = self._saved_plots[row].get("name", "Unnamed Plot")
                item.setText(old_name)
            return

        # Get the row index to find the corresponding plot config
        row = self.saved_plots_list.row(item)
        if 0 <= row < len(self._saved_plots):
            old_name = self._saved_plots[row].get("name", "Unnamed Plot")

            # Check if name is different
            if new_name != old_name:
                # Update the plot config
                self._saved_plots[row]["name"] = new_name

                # Save to HDF5 file using the same method as other plot operations
                if self._current_csv_group_path and self.model and self.model.filepath:
                    try:
                        with h5py.File(self.model.filepath, "r+") as h5:
                            if self._current_csv_group_path in h5:
                                grp = h5[self._current_csv_group_path]
                                if isinstance(grp, h5py.Group):
                                    plots_json = json.dumps(self._saved_plots)
                                    grp.attrs["saved_plots"] = plots_json
                        self.statusBar().showMessage(f"Renamed plot to '{new_name}'", 3000)
                    except Exception as e:
                        self.statusBar().showMessage(f"Error saving renamed plot: {e}", 5000)
                        # Revert the name on error
                        item.setText(old_name)
                        self._saved_plots[row]["name"] = old_name

    def _apply_filtered_indices_to_data(
        self,
        col_data_dict: dict[str, np.ndarray],
        filtered_indices: list[int] | None,
        start_row: int = 0,
        end_row: int = -1
    ) -> dict[str, np.ndarray]:
        """Apply filtered indices to column data with bounds checking.

        Args:
            col_data_dict: Dictionary of column name -> numpy array
            filtered_indices: List of row indices to include, or None
            start_row: Starting row for backward compatibility (used if filtered_indices is None)
            end_row: Ending row for backward compatibility (used if filtered_indices is None)

        Returns:
            Dictionary with filtered data
        """
        result = {}
        for col_name, col_array in col_data_dict.items():
            if isinstance(col_array, np.ndarray) and len(col_array) > 0:
                if filtered_indices is not None:
                    # Use the stored filtered indices (handles non-contiguous filtering)
                    filtered_indices_array = np.array(filtered_indices, dtype=int)
                    # Filter out any indices that are out of bounds for the current data
                    valid_indices = filtered_indices_array[filtered_indices_array < len(col_array)]
                    if len(valid_indices) > 0:
                        result[col_name] = col_array[valid_indices]
                    else:
                        result[col_name] = np.array([])
                elif end_row >= 0 and end_row < len(col_array):
                    # Backward compatibility: use row range
                    result[col_name] = col_array[start_row : end_row + 1]
                else:
                    result[col_name] = col_array[start_row:]
            else:
                result[col_name] = col_array
        return result

    def _apply_saved_plot(self, item: QListWidgetItem | None = None) -> None:
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
        x_idx = plot_config.get("x_col_idx")  # Can be None for single-column plots
        y_idxs = plot_config.get("y_col_idxs", [])
        # Ensure y_idxs is a list (handle legacy configs where it might be an integer)
        if isinstance(y_idxs, int):
            y_idxs = [y_idxs]
        filtered_indices = plot_config.get("filtered_indices")
        start_row = plot_config.get("start_row", 0)
        end_row = plot_config.get("end_row", -1)

        if not y_idxs:
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
        if (x_idx is not None and x_idx >= len(headers)) or any(y_idx >= len(headers) for y_idx in y_idxs):
            QMessageBox.warning(
                self, "Invalid Columns", "Plot configuration references invalid column indices."
            )
            return

        try:
            x_name = headers[x_idx] if x_idx is not None else "Point"
            y_names = [headers[i] for i in y_idxs]
        except Exception:
            QMessageBox.warning(
                self, "Plot Error", "Failed to resolve column headers for plotting."
            )
            return

        # Read column data directly from HDF5 for plotting (don't use table's lazy-loading cache)
        columns_to_read = y_names if x_idx is None else [x_name] + y_names
        col_data = self._read_csv_columns(self._current_csv_group_path, columns_to_read)
        if not col_data:
            QMessageBox.warning(self, "Plot Error", "Failed to read column data from HDF5.")
            return

        # Handle non-array data
        for name in list(col_data.keys()):
            if not isinstance(col_data[name], np.ndarray):
                col_data[name] = np.array([col_data[name]])

        # Apply filtered indices with bounds checking
        col_data = self._apply_filtered_indices_to_data(col_data, filtered_indices, start_row, end_row)

        if not any(name in col_data for name in y_names):
            QMessageBox.warning(self, "Plot Error", "Failed to get column data for plotting.")
            return

        # Plot the data
        try:
            # Get plot options from configuration
            plot_options = plot_config.get("plot_options", {})

            # Process X-axis data using helper method
            x_arr, x_num, x_is_string, xaxis_datetime, min_len = self._process_x_axis_data(
                x_idx, col_data, y_names, x_name, plot_options
            )

            if min_len <= 0:
                QMessageBox.warning(self, "Plot Error", "No data to plot.")
                return

            # Clear previous plot
            self.plot_figure.clear()

            # Get plot options to check for dark background
            use_dark = plot_options.get("dark_background", False)

            # Create subplot and apply style
            ax = self.plot_figure.add_subplot(111)
            self._apply_plot_style(self.plot_figure, ax, use_dark)

            # Disable offset notation on axes
            ax.ticklabel_format(useOffset=False)

            # Get series styling options
            series_styles = plot_options.get("series", {})
            series_visibility = plot_options.get("series_visibility", {})

            any_plotted = False
            for y_name in y_names:
                if y_name not in col_data:
                    continue
                y_arr = col_data[y_name].ravel()[:min_len]
                y_num = pd.to_numeric(pd.Series(y_arr), errors="coerce").astype(float).to_numpy()
                valid = np.isfinite(x_num) & np.isfinite(y_num)
                if valid.any():
                    series_opts = series_styles.get(y_name, {})
                    any_plotted = self._plot_series_with_options(
                        ax, x_num, y_num, valid, y_name, series_opts, any_plotted
                    )

            if not any_plotted:
                QMessageBox.information(self, "Plot", "No valid numeric data found to plot.")
                return

            # Adjust title for row range if needed
            title_suffix = ""
            if start_row > 0 or end_row < len(self._csv_data_dict.get(x_name, [])) - 1:
                title_suffix = f" (rows {start_row}-{end_row})"

            # Create modified plot_config with title suffix
            modified_config = plot_config.copy()
            if not plot_options.get("title", "").strip():
                modified_config["name"] = plot_config.get("name", "Plot") + title_suffix

            # Format x-axis (datetime or categorical strings)
            self._format_xaxis(ax, self.plot_figure, xaxis_datetime, x_is_string, x_arr, min_len, plot_options)

            # Apply axis limits
            self._apply_axis_limits(ax, plot_options)

            # Apply labels, fonts, grid, legend, log scale, and reference lines
            self._apply_plot_labels_and_formatting(
                ax, self.plot_figure, x_name, y_names, modified_config, plot_options, use_dark
            )

            # Apply saved visibility state AFTER formatting (so legend is already created)
            self._apply_plot_visibility_state(ax, y_names, series_visibility)

            # Refresh canvas
            self.plot_canvas.draw()

            # Switch to Plot tab
            self.bottom_tabs.setCurrentIndex(1)

            self.statusBar().showMessage(
                f"Applied plot: {plot_config.get('name', 'Unnamed')}", 3000
            )

        except Exception as exc:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot data:\n{exc}")

    def _export_plot_to_file(self, plot_config: dict, filepath: str) -> tuple[bool, str]:
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
            # Ensure y_idxs is a list (handle legacy configs)
            if isinstance(y_idxs, int):
                y_idxs = [y_idxs]

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

            # Validate column indices (x_idx can be None for single-column plots)
            if x_idx is not None and x_idx >= len(headers):
                return False, "Invalid X column index"
            if not all(idx < len(headers) for idx in y_idxs):
                return False, "Invalid Y column indices"

            # Get column names
            x_name = headers[x_idx] if x_idx is not None else "Point"
            y_names = [headers[i] for i in y_idxs]

            # Read column data (only for columns that exist)
            columns_to_read = y_names if x_idx is None else [x_name] + y_names
            col_data = self._read_csv_columns(group_path, columns_to_read)
            if not col_data:
                return False, "Failed to read column data"

            if not any(name in col_data for name in y_names):
                return False, "Y column data not found"

            # For None x_idx, x_name won't be in col_data, which is expected
            if x_idx is not None and x_name not in col_data:
                return False, "X column data not found"

            # Apply filtering if specified in plot config
            # This ensures filtered data is used during export (respects filters from when plot was saved)
            filtered_indices = plot_config.get("filtered_indices")
            start_row = plot_config.get("start_row", 0)
            end_row = plot_config.get("end_row", -1)

            # Apply filtered indices with bounds checking
            col_data = self._apply_filtered_indices_to_data(col_data, filtered_indices, start_row, end_row)

            # Get plot options
            plot_options = plot_config.get("plot_options", {})
            figwidth = plot_options.get("figwidth", 8.0)
            figheight = plot_options.get("figheight", 6.0)
            dpi = plot_options.get("dpi", 100)
            use_dark = plot_options.get("dark_background", False)

            # Create figure with specified size
            fig = Figure(figsize=(figwidth, figheight), dpi=dpi)

            # Create subplot and apply style
            ax = fig.add_subplot(111)
            self._apply_plot_style(fig, ax, use_dark)

            # Disable offset notation on axes
            ax.ticklabel_format(useOffset=False)

            # Process X-axis data using helper method
            x_arr, x_num, x_is_string, xaxis_datetime, min_len = self._process_x_axis_data(
                x_idx, col_data, y_names, x_name, plot_options
            )

            if min_len <= 0:
                return False, "No data to plot"

            # Plot series with smoothing support
            series_styles = plot_options.get("series", {})
            series_visibility = plot_options.get("series_visibility", {})
            any_plotted = False

            for y_name in y_names:
                if y_name not in col_data:
                    continue
                y_arr = col_data[y_name].ravel()[:min_len]
                y_num = pd.to_numeric(pd.Series(y_arr), errors="coerce").astype(float).to_numpy()
                valid = np.isfinite(x_num) & np.isfinite(y_num)
                if valid.any():
                    series_opts = series_styles.get(y_name, {})
                    any_plotted = self._plot_series_with_options(
                        ax, x_num, y_num, valid, y_name, series_opts, any_plotted
                    )

            if not any_plotted:
                return False, "No valid numeric data to plot"

            # Date formatting (must be done before labels/formatting)
            if xaxis_datetime:
                display_format = plot_options.get("datetime_display_format", "").strip()
                if display_format:
                    ax.xaxis.set_major_formatter(DateFormatter(display_format))
                else:
                    ax.xaxis.set_major_locator(AutoDateLocator())
                fig.autofmt_xdate()

            # Apply axis limits
            self._apply_axis_limits(ax, plot_options)

            # Apply labels, fonts, grid, legend, log scale, and reference lines
            # Note: interactive_legend=False for export to avoid event handler issues
            self._apply_plot_labels_and_formatting(
                ax, fig, x_name, y_names, plot_config, plot_options, use_dark, interactive_legend=False
            )

            # Apply saved visibility state AFTER formatting (so legend is already created)
            self._apply_plot_visibility_state(ax, y_names, series_visibility)

            # Save to file
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

            return True, ""

        except Exception as e:
            error_msg = f"Error exporting plot: {e}"
            print(error_msg)
            traceback.print_exc()
            return False, str(e)

    def _export_all_plots(self) -> None:
        """Export all saved plots to a selected directory."""
        if not self._saved_plots:
            QMessageBox.information(self, "No Plots", "There are no saved plots to export.")
            return

        # Ask user to select output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Export Plots",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )

        if not output_dir:
            return  # User cancelled

        # Create progress dialog
        progress = QProgressDialog(
            "Exporting plots...",
            "Cancel",
            0,
            len(self._saved_plots),
            self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        # Export each plot
        exported_count = 0
        failed_plots = []

        for i, plot_config in enumerate(self._saved_plots):
            if progress.wasCanceled():
                break

            plot_name = plot_config.get("name", f"plot_{i+1}")
            progress.setLabelText(f"Exporting: {plot_name}")
            progress.setValue(i)
            QApplication.processEvents()

            # Sanitize filename (remove invalid characters)
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in plot_name)
            safe_name = safe_name.strip()
            if not safe_name:
                safe_name = f"plot_{i+1}"

            # Get export format from plot options, default to PNG
            plot_options = plot_config.get("plot_options", {})
            export_format = plot_options.get("export_format", "png").lower()
            if export_format not in ["png", "pdf", "svg", "jpg", "jpeg"]:
                export_format = "png"

            filepath = os.path.join(output_dir, f"{safe_name}.{export_format}")

            # If file exists, add number suffix
            counter = 1
            while os.path.exists(filepath):
                filepath = os.path.join(output_dir, f"{safe_name}_{counter}.{export_format}")
                counter += 1

            # Export the plot
            success, error_msg = self._export_plot_to_file(plot_config, filepath)

            if success:
                exported_count += 1
            else:
                failed_plots.append((plot_name, error_msg))

        progress.setValue(len(self._saved_plots))
        progress.close()

        # Show results
        if failed_plots:
            failure_details = "\n".join([f"• {name}: {error}" for name, error in failed_plots])
            QMessageBox.warning(
                self,
                "Export Complete with Errors",
                f"Exported {exported_count} of {len(self._saved_plots)} plots to:\n{output_dir}\n\n"
                f"Failed plots:\n{failure_details}"
            )
        else:
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {exported_count} plot(s) to:\n{output_dir}"
            )

        self.statusBar().showMessage(f"Exported {exported_count} plot(s) to {output_dir}", 5000)

    def _delete_plot_config(self) -> None:
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

    def _edit_plot_options_dialog(self) -> None:
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

            # Check if filters or sort changed - if so, recalculate filtered_indices
            old_filters = plot_config.get("csv_filters", [])
            old_sort = plot_config.get("csv_sort", [])
            new_filters = updated_config.get("csv_filters", [])
            new_sort = updated_config.get("csv_sort", [])

            if old_filters != new_filters or old_sort != new_sort:
                # Recalculate filtered_indices based on new filters and sort
                self._recalculate_plot_filtered_indices(updated_config)

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

    def _recalculate_plot_filtered_indices(self, plot_config: dict) -> None:
        """Recalculate filtered_indices for a plot based on its filter and sort settings.

        Args:
            plot_config: Plot configuration dictionary to update
        """
        if not self._csv_data_dict or not self._csv_column_names:
            return

        # Get filters and sort from plot config
        csv_filters = plot_config.get("csv_filters", [])
        csv_sort = plot_config.get("csv_sort", [])

        # Get total row count
        max_rows = max(len(self._csv_data_dict[col]) for col in self._csv_data_dict) if self._csv_data_dict else 0

        if max_rows == 0:
            plot_config["filtered_indices"] = None
            return

        # Start with all rows
        filtered_indices = np.arange(max_rows)

        # Apply filters
        if csv_filters:
            for col_name, operator, value_str in csv_filters:
                if col_name not in self._csv_data_dict:
                    continue

                col_data = self._csv_data_dict[col_name]

                # Only filter on rows we haven't already filtered out
                valid_mask = np.ones(len(filtered_indices), dtype=bool)

                for i, row_idx in enumerate(filtered_indices):
                    if row_idx >= len(col_data):
                        valid_mask[i] = False
                        continue

                    cell_value = col_data[row_idx]

                    # Apply filter using helper methods
                    try:
                        if operator in ("=", "==", "!=", "<", "<=", ">", ">="):
                            # Use comparison helper
                            matches = self._compare_values(cell_value, operator, value_str)
                        elif operator == "contains":
                            matches = value_str.lower() in str(cell_value).lower()
                        elif operator == "starts with":
                            matches = str(cell_value).lower().startswith(value_str.lower())
                        elif operator == "ends with":
                            matches = str(cell_value).lower().endswith(value_str.lower())
                        else:
                            matches = True

                        valid_mask[i] = matches
                    except (ValueError, TypeError):
                        # If comparison fails, exclude the row
                        valid_mask[i] = False

                # Update filtered_indices to only include rows that passed this filter
                filtered_indices = filtered_indices[valid_mask]

        # Apply sort
        if csv_sort and len(filtered_indices) > 0:
            # Build a structured array for sorting
            sort_data = []
            for col_name, order in csv_sort:
                if col_name in self._csv_data_dict:
                    col_data = self._csv_data_dict[col_name]
                    # Extract data for filtered rows
                    sort_column = np.array([col_data[i] if i < len(col_data) else "" for i in filtered_indices])
                    sort_data.append((col_name, sort_column))

            if sort_data:
                # Sort using lexsort (sorts by last column first, then previous, etc.)
                sort_arrays = []
                for col_name, order in reversed(csv_sort):
                    for stored_name, stored_array in sort_data:
                        if stored_name == col_name:
                            # Normalize order to boolean: True = ascending, False = descending
                            # Handle various formats: "asc"/"desc", "Asc"/"Desc", True/False
                            if isinstance(order, bool):
                                is_ascending = order
                            elif isinstance(order, str):
                                is_ascending = order.lower() not in ('desc', 'descending')
                            else:
                                is_ascending = True  # Default to ascending

                            # Convert to numeric if possible for better sorting
                            try:
                                numeric_array = pd.to_numeric(pd.Series(stored_array), errors='coerce')
                                if not is_ascending:
                                    sort_arrays.append(-numeric_array.to_numpy())
                                else:
                                    sort_arrays.append(numeric_array.to_numpy())
                            except Exception:
                                # Fall back to string sorting
                                str_array = np.array([str(x) for x in stored_array])
                                # For string sorting, we'll handle descending after lexsort
                                sort_arrays.append(str_array)
                            break

                if sort_arrays:
                    sort_idx = np.lexsort(tuple(sort_arrays))
                    filtered_indices = filtered_indices[sort_idx]

        # Update plot config with new filtered indices
        if len(filtered_indices) > 0:
            plot_config["filtered_indices"] = filtered_indices.tolist()
            plot_config["start_row"] = int(filtered_indices[0])
            plot_config["end_row"] = int(filtered_indices[-1])
        else:
            plot_config["filtered_indices"] = []
            plot_config["start_row"] = 0
            plot_config["end_row"] = 0

    def _copy_plot_json_to_clipboard(self):
        """Copy the selected plot's JSON configuration to clipboard."""
        current_row = self.saved_plots_list.currentRow()
        if current_row < 0 or current_row >= len(self._saved_plots):
            return

        plot_config = self._saved_plots[current_row]

        try:
            # Convert plot config to JSON with nice formatting
            json_str = json.dumps(plot_config, indent=2, default=str)

            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(json_str)

            plot_name = plot_config.get('name', 'Unnamed')
            self.statusBar().showMessage(
                f"Copied JSON for '{plot_name}' to clipboard", 3000
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Copy Failed",
                f"Failed to copy plot JSON to clipboard.\n\nError: {e}"
            )

    def _on_attrs_context_menu(self, point):
        """Show context menu for attributes table."""
        item = self.attrs_table.itemAt(point)
        if item is None:
            return

        row = item.row()
        col = item.column()

        menu = QMenu(self)

        # Only show copy if a value cell is clicked (column 1)
        if col == 1:
            act_copy = menu.addAction("Copy Value")
        else:
            act_copy = None

        # Always show paste option for value cells
        if col == 1:
            act_paste = menu.addAction("Paste Value")
        else:
            act_paste = None

        if act_copy is None and act_paste is None:
            return  # No actions available

        global_pos = self.attrs_table.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)

        if chosen == act_copy:
            # Copy full attribute value from HDF5 file (not truncated display text)
            name_item = self.attrs_table.item(row, 0)
            if not name_item:
                return

            attr_name = name_item.text()

            # Get the current object path from selection
            sel = self.tree.selectionModel().selectedIndexes()
            if not sel:
                return

            index = sel[0].sibling(sel[0].row(), 0)
            item = self.model.itemFromIndex(index)
            if item is None:
                return

            obj_path = item.data(self.model.ROLE_PATH)
            if not obj_path:
                return

            # Read the full attribute value from the HDF5 file
            fpath = self.model.filepath
            if not fpath:
                return

            try:
                with h5py.File(fpath, "r") as h5:
                    obj = h5[obj_path]
                    if attr_name in obj.attrs:
                        attr_value = obj.attrs[attr_name]
                        # Convert to string for clipboard
                        if isinstance(attr_value, (np.ndarray, list)):
                            # For arrays/lists, use repr to get full representation
                            value_str = repr(attr_value)
                        else:
                            value_str = str(attr_value)

                        clipboard = QApplication.clipboard()
                        clipboard.setText(value_str)
                        self.statusBar().showMessage("Full attribute value copied to clipboard", 2000)
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to copy attribute: {exc}", 3000)
        elif chosen == act_paste:
            # Paste value from system clipboard
            clipboard = QApplication.clipboard()
            clipboard_text = clipboard.text().strip()
            if not clipboard_text:
                QMessageBox.warning(self, "Paste", "Clipboard is empty.")
                return

            # Get attribute name
            name_item = self.attrs_table.item(row, 0)
            if not name_item:
                return

            attr_name = name_item.text()

            # Get the current object path from selection
            sel = self.tree.selectionModel().selectedIndexes()
            if not sel:
                QMessageBox.warning(self, "Paste", "No item selected in tree.")
                return

            index = sel[0].sibling(sel[0].row(), 0)
            item = self.model.itemFromIndex(index)
            if item is None:
                return

            obj_path = item.data(self.model.ROLE_PATH)
            if not obj_path:
                return

            # Update the attribute in the HDF5 file
            fpath = self.model.filepath
            if not fpath:
                QMessageBox.warning(self, "Paste", "No HDF5 file loaded.")
                return

            try:
                with h5py.File(fpath, "r+") as h5:
                    obj = h5[obj_path]

                    # Simply store as string - no parsing or conversion
                    obj.attrs[attr_name] = clipboard_text

                # Update the table display
                value_item = self.attrs_table.item(row, 1)
                if value_item:
                    value_item.setText(clipboard_text)

                self.statusBar().showMessage(f"Updated attribute '{attr_name}' in HDF5 file", 3000)

                # Special handling for saved_plots attribute - recalculate filtered indices
                if attr_name == "saved_plots" and obj_path == self._current_csv_group_path:
                    try:
                        # Parse the JSON to get the updated plots
                        import json
                        plots = json.loads(clipboard_text)
                        if isinstance(plots, list):
                            # Update each plot's filtered_indices based on its filters/sort
                            for plot_config in plots:
                                if plot_config.get("csv_filters") or plot_config.get("csv_sort"):
                                    self._recalculate_plot_filtered_indices(plot_config)

                            # Save the updated plots back to HDF5
                            with h5py.File(fpath, "r+") as h5:
                                if obj_path in h5:
                                    h5[obj_path].attrs[attr_name] = json.dumps(plots)

                            # Reload the plots in the UI
                            self._saved_plots = plots
                            self._refresh_saved_plots_list()

                            self.statusBar().showMessage(
                                f"Updated {len(plots)} plot(s) and recalculated filtered indices", 3000
                            )
                    except Exception as e:
                        print(f"Warning: Could not recalculate plot filtered indices: {e}")

                # Refresh the display to reflect any changes
                kind = item.data(self.model.ROLE_KIND)
                if kind == "dataset":
                    self.preview_dataset(obj_path)
                elif kind == "group":
                    self.preview_group(obj_path)
                elif kind == "attr":
                    # If we're on an attr node, refresh its parent
                    parent_path = item.data(self.model.ROLE_PATH)
                    if parent_path:
                        # Determine if parent is dataset or group
                        try:
                            with h5py.File(fpath, "r") as h5:
                                parent_obj = h5[parent_path]
                                if isinstance(parent_obj, h5py.Dataset):
                                    self.preview_dataset(parent_path)
                                elif isinstance(parent_obj, h5py.Group):
                                    self.preview_group(parent_path)
                        except Exception:
                            pass

            except Exception as exc:
                QMessageBox.critical(self, "Paste Failed", f"Failed to update attribute:\n{exc}")

    def _duplicate_plot_config(self):
        """Duplicate the selected plot configuration."""
        current_row = self.saved_plots_list.currentRow()
        if current_row < 0 or current_row >= len(self._saved_plots):
            return

        plot_config = self._saved_plots[current_row]
        plot_name = plot_config.get("name", "Unnamed Plot")

        # Create a deep copy of the plot config
        import copy
        duplicated_config = copy.deepcopy(plot_config)

        # Update the name to indicate it's a copy
        duplicated_config["name"] = f"{plot_name} (copy)"
        duplicated_config["timestamp"] = time.time()

        # Add to the end of the list
        self._saved_plots.append(duplicated_config)

        # Save to HDF5
        self._save_plot_configs_to_hdf5()

        # Refresh list widget
        self._refresh_saved_plots_list()

        # Select the new duplicated plot
        self.saved_plots_list.setCurrentRow(len(self._saved_plots) - 1)

        self.statusBar().showMessage(f"Duplicated plot: {plot_name}", 3000)

    def _on_saved_plots_context_menu(self, point):
        """Show context menu for saved plots list."""
        item = self.saved_plots_list.itemAt(point)

        menu = QMenu(self)

        # Actions that require a selected item
        if item is not None:
            act_duplicate = menu.addAction("Duplicate Plot")
            act_copy_json = menu.addAction("Copy JSON to Clipboard")
            menu.addSeparator()
            act_delete = menu.addAction("Delete Plot")
            menu.addSeparator()

        # Export all plots action (always available if plots exist)
        act_export_all = None
        if len(self._saved_plots) > 0:
            act_export_all = menu.addAction("Export All Plots...")

        # If no actions available, don't show menu
        if menu.isEmpty():
            return

        global_pos = self.saved_plots_list.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)

        if item is not None:
            if chosen == act_duplicate:
                self._duplicate_plot_config()
            elif chosen == act_copy_json:
                self._copy_plot_json_to_clipboard()
            elif chosen == act_delete:
                self._delete_plot_config()

        if chosen == act_export_all:
            self._export_all_plots()


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
