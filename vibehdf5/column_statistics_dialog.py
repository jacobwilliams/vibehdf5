""" "
Dialog for displaying column statistics.
"""

import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .table_copy_mixin import TableCopyMixin


class ColumnStatisticsDialog(TableCopyMixin, QDialog):
    """Dialog for displaying column statistics."""

    def __init__(
        self,
        column_names: list[str],
        data_dict: dict[str, np.ndarray],
        filtered_indices: np.ndarray | None,
        parent: QWidget | None = None,
    ) -> None:
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

        self.column_names: list[str] = column_names
        """List of column names to display statistics for"""
        self.data_dict: dict[str, np.ndarray] = data_dict
        """Dictionary mapping column names to data arrays"""
        self.filtered_indices: np.ndarray | None = filtered_indices
        """Array of filtered row indices, or None for all rows"""

        layout = QVBoxLayout(self)

        # Info label
        if filtered_indices is not None and len(filtered_indices) > 0:
            total_rows = max(len(data_dict[col]) for col in data_dict if col in column_names)
            info_text = (
                f"Statistics for {len(filtered_indices)} filtered rows (out of {total_rows} total)"
            )
        else:
            total_rows = max(len(data_dict[col]) for col in data_dict if col in column_names)
            info_text = f"Statistics for all {total_rows} rows"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(info_label)

        # Statistics table (named 'table' for TableCopyMixin compatibility)
        self.table: QTableWidget = QTableWidget(self)
        """Table widget for displaying column statistics"""
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        # Calculate and display statistics
        self._calculate_statistics()

        # Enable copy functionality
        self.setup_table_copy()

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _calculate_statistics(self):
        """Calculate statistics for all columns."""
        # Define statistics to calculate
        stat_labels = ["Count", "Min", "Max", "Mean", "Median", "Std Dev", "Sum", "Unique Values"]

        # Set up table dimensions
        self.table.setRowCount(len(stat_labels))
        self.table.setColumnCount(len(self.column_names))
        self.table.setHorizontalHeaderLabels(self.column_names)
        self.table.setVerticalHeaderLabels(stat_labels)

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
                    stats["Min"] = (
                        f"{numeric_series.min():.6g}" if not numeric_series.isna().all() else "N/A"
                    )
                    stats["Max"] = (
                        f"{numeric_series.max():.6g}" if not numeric_series.isna().all() else "N/A"
                    )
                    stats["Mean"] = (
                        f"{numeric_series.mean():.6g}" if not numeric_series.isna().all() else "N/A"
                    )
                    stats["Median"] = (
                        f"{numeric_series.median():.6g}"
                        if not numeric_series.isna().all()
                        else "N/A"
                    )
                    stats["Std Dev"] = (
                        f"{numeric_series.std():.6g}" if not numeric_series.isna().all() else "N/A"
                    )
                    stats["Sum"] = (
                        f"{numeric_series.sum():.6g}" if not numeric_series.isna().all() else "N/A"
                    )
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
                    self.table.setItem(row_idx, col_idx, item)

            except Exception as e:
                # On error, fill with N/A
                for row_idx in range(len(stat_labels)):
                    item = QTableWidgetItem("Error")
                    item.setToolTip(str(e))
                    self.table.setItem(row_idx, col_idx, item)

        # Resize columns to content
        self.table.resizeColumnsToContents()
