"""
Dialog for configuring multi-column sorting.
"""

from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QScrollArea,
    QWidget,
    QFrame,
    QDialogButtonBox,
)


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
        add_btn.setToolTip(
            "Add a new column to sort by (columns are sorted in order from top to bottom)"
        )
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
