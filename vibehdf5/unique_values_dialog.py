"""
Dialog for displaying unique values in a column.
"""

from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class UniqueValuesDialog(QDialog):
    """Dialog for displaying unique values in a column."""

    def __init__(self, column_name: str, unique_values: list, parent=None):
        """Initialize the unique values dialog.

        Args:
            column_name (str): Name of the column being displayed
            unique_values (list): List of unique values to display
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(f"Unique Values - {column_name}")
        self.resize(400, 500)

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            f"Unique values in column '{column_name}' ({len(unique_values)} unique):"
        )
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
