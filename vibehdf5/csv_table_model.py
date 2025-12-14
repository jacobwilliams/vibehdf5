"""
Table model for displaying CSV data in a QTableView.
"""

import numpy as np
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt


class CSVTableModel(QAbstractTableModel):
    """Table model for displaying CSV data in a QTableView."""

    def __init__(
        self,
        data_dict: dict[str, np.ndarray],
        col_names: list[str],
        row_indices: list[int] | None = None,
        parent=None,
    ):
        """
        Initialize the table model for QTableView.
        Args:
            data_dict: Dictionary mapping column names to data arrays.
            col_names: List of column names in display order.
            row_indices: Optional list/array of row indices to display (for filtering/sorting).
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self._data_dict = data_dict
        self._col_names = col_names
        # Cached list of column data references (aligned with _col_names)
        self._col_data_refs: list = []
        self._rebuild_column_refs()
        self._row_indices: np.ndarray | None = None
        if row_indices is not None:
            self._row_indices = np.array(row_indices)
            self._row_count = len(self._row_indices)
        elif col_names and col_names[0] in data_dict:
            self._row_indices = None
            self._row_count = len(data_dict[col_names[0]])
        else:
            self._row_indices = None
            self._row_count = 0

    def set_row_indices(self, indices, total_rows=None):
        """
        Update the model to display only the specified row indices (for filtering/sorting).
        This method updates the internal row count and emits a signal to refresh the QTableView.

        Args:
            indices: List or array of row indices to display. If None, all rows are shown.
            total_rows: Optional total number of rows to display if indices is None.
        """
        # Use more precise reset notifications when we change the model
        self.beginResetModel()
        try:
            if indices is None:
                self._row_indices = None
                self._row_count = total_rows if total_rows is not None else 0
            else:
                # Always use a NumPy array for indexing
                self._row_indices = np.array(indices)
                self._row_count = len(self._row_indices)
            # Rebuild cached references in case underlying dict changed
            self._rebuild_column_refs()
        finally:
            self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        """
        Return the number of rows currently displayed in the QTableView.
        This reflects filtering/sorting if row_indices is set.
        """
        return self._row_count

    def columnCount(self, parent=QModelIndex()):
        """
        Return the number of columns in the table.
        """
        return len(self._col_names)

    def data(self, index, role=Qt.DisplayRole):
        """
        Return the data to display for a given cell in QTableView.
        Handles filtering/sorting via row_indices, and decodes bytes as needed.

        Args:
            index: QModelIndex specifying the cell location.
            role: Qt.ItemDataRole specifying the type of data requested (default: Qt.DisplayRole).

        Returns:
            The data to display for the specified cell, or None if not valid.
        """
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            row = index.row()
            col = index.column()
            # Access cached column data reference (faster than dict lookups)
            try:
                col_data = self._col_data_refs[col]
            except Exception:
                return ""
            # Use filtered indices if present
            if self._row_indices is not None:
                if row < len(self._row_indices):
                    data_idx = int(self._row_indices[row])
                else:
                    return ""
            else:
                data_idx = row
            if data_idx < len(col_data):
                val = col_data[data_idx]
                # Fast-paths for common display types to avoid repeated conversions
                if val is None:
                    return ""
                if isinstance(val, str):
                    return val
                if isinstance(val, bytes):
                    return val.decode("utf-8", errors="replace")
                # numpy scalar fast-paths
                if isinstance(val, (np.integer, np.floating)):
                    return str(val)
                # Fallback to str() for other types
                return str(val)
            return ""
        return None

    def _rebuild_column_refs(self) -> None:
        """Rebuild cached list of column data references aligned with `_col_names`.

        This avoids repeated dictionary lookups from `data()` which is called
        very frequently by Qt while rendering the view.
        """
        refs = []
        for name in self._col_names:
            refs.append(self._data_dict.get(name, np.array([], dtype=object)))
        self._col_data_refs = refs

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """
        Return the header label for columns and rows in QTableView.

        Args:
            section: Integer index of the column or row.
            orientation: Qt.Orientation specifying horizontal (columns) or vertical (rows).
            role: Qt.ItemDataRole specifying the type of data requested (default: Qt.DisplayRole).

        Returns:
            The header label for the specified section and orientation, or None if not valid.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section < len(self._col_names):
                    return self._col_names[section]
            elif orientation == Qt.Vertical:
                return str(section + 1)
        return None

    def sort(self, column, order):
        """
        Sorting is handled externally via filtered indices and set_row_indices.
        This method is a no-op for compatibility.
        """
        pass
