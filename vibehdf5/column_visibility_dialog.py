
from qtpy.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QDialogButtonBox, QScrollArea, QWidget, QFrame


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


