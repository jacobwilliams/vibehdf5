"""
Dialog for selecting which columns to display in the CSV table.
"""

import fnmatch

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class ColumnVisibilityDialog(QDialog):
    """Dialog for selecting which columns to display in the CSV table."""

    def __init__(
        self, column_names: list[str], visible_columns: list[str] | None = None, parent=None
    ):
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
        self.visible_columns = (
            visible_columns if visible_columns is not None else column_names.copy()
        )

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

        # Search field for filtering columns
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_layout.addWidget(search_label)

        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Type to filter columns...")
        self.search_field.setClearButtonEnabled(True)
        self.search_field.textChanged.connect(self._filter_columns)
        search_layout.addWidget(self.search_field)

        layout.addLayout(search_layout)

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

    def _on_show_all_toggled(self, checked: bool):
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

    def _filter_columns(self, text: str):
        """Filter the column checkboxes based on the search text, supporting wildcards (*, ?).

        Args:
            text: The search text entered by the user. Supports wildcards (e.g., 'temp*', '*id*').
        """
        pattern = text.lower().strip()
        for checkbox in self.column_checkboxes:
            col_name = checkbox.text().lower()
            if not pattern:
                checkbox.setVisible(True)
            elif "*" in pattern or "?" in pattern:
                checkbox.setVisible(fnmatch.fnmatch(col_name, pattern))
            else:
                checkbox.setVisible(pattern in col_name)

    def get_visible_columns(self):
        """Return list of selected column names."""
        return [cb.text() for cb in self.column_checkboxes if cb.isChecked()]
