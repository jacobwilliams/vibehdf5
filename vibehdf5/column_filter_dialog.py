
from qtpy.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, QDialogButtonBox, QScrollArea, QWidget, QFrame


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

