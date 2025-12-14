"""
Mixin class for table copy functionality in dialogs.
"""

from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QAbstractItemView, QApplication, QMenu, QStyle


class TableCopyMixin:
    """Mixin class providing copy functionality for QTableWidget in dialogs.

    Classes using this mixin must have a 'table' attribute that is a QTableWidget.
    """

    def setup_table_copy(self):
        """Enable copy functionality for the table.

        Call this method after creating self.table to enable context menu and keyboard shortcuts.
        Configures the table to support:
        - Column selection by clicking column headers
        - Row selection by clicking row headers
        - Individual cell selection by clicking cells
        """
        if not hasattr(self, 'table'):
            raise AttributeError("TableCopyMixin requires a 'table' attribute (QTableWidget)")

        # Enable selection of rows, columns, and individual cells
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)

        # Allow clicking headers to select entire rows/columns
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.verticalHeader().setSectionsClickable(True)

        # Connect header clicks for column/row selection
        self.table.horizontalHeader().sectionClicked.connect(self._on_header_column_clicked)
        self.table.verticalHeader().sectionClicked.connect(self._on_header_row_clicked)

        # Enable context menu
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)

    def _on_header_column_clicked(self, logical_index):
        """Handle column header click to select entire column."""
        self.table.selectColumn(logical_index)

    def _on_header_row_clicked(self, logical_index):
        """Handle row header click to select entire row."""
        self.table.selectRow(logical_index)

    def _on_table_context_menu(self, point):
        """Show context menu for table with copy options."""
        menu = QMenu(self)
        style = self.style()

        # Copy actions (tab-separated)
        act_copy = menu.addAction("Copy")
        act_copy.setShortcut("Ctrl+C")
        act_copy.setIcon(style.standardIcon(QStyle.SP_FileDialogDetailedView))

        act_copy_with_headers = menu.addAction("Copy with Headers")
        act_copy_with_headers.setIcon(style.standardIcon(QStyle.SP_FileDialogDetailedView))

        menu.addSeparator()

        # CSV copy actions (comma-separated)
        act_copy_csv = menu.addAction("Copy as CSV")
        act_copy_csv.setIcon(style.standardIcon(QStyle.SP_FileDialogDetailedView))

        act_copy_csv_with_headers = menu.addAction("Copy as CSV with Headers")
        act_copy_csv_with_headers.setIcon(style.standardIcon(QStyle.SP_FileDialogDetailedView))

        # Show menu
        global_pos = self.table.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)

        if chosen == act_copy:
            self._copy_table_selection(include_headers=False, separator="\t")
        elif chosen == act_copy_with_headers:
            self._copy_table_selection(include_headers=True, separator="\t")
        elif chosen == act_copy_csv:
            self._copy_table_selection(include_headers=False, separator=",")
        elif chosen == act_copy_csv_with_headers:
            self._copy_table_selection(include_headers=True, separator=",")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for copy functionality.

        Implements Ctrl+C for copying selected table cells.
        """
        if event.matches(QKeySequence.Copy):
            if hasattr(self, 'table') and self.table.hasFocus():
                self._copy_table_selection(include_headers=False, separator="\t")
                event.accept()
                return

        # Call parent implementation for other keys
        super().keyPressEvent(event)

    def _copy_table_selection(self, include_headers: bool = False, separator: str = "\t"):
        """Copy selected table cells to clipboard.

        Supports copying:
        - Individual cells
        - Cell ranges
        - Entire columns
        - Mixed selections

        Args:
            include_headers: If True, include column headers and row labels in the copied data
            separator: Field separator character ("\t" for tab, "," for CSV)
        """
        if not hasattr(self, 'table'):
            return

        selection_model = self.table.selectionModel()
        if not selection_model:
            return

        # Determine which columns and rows are selected
        selected_columns = set()
        selected_rows = set()

        # Check if entire columns are selected
        for col_idx in range(self.table.columnCount()):
            if selection_model.isColumnSelected(col_idx):
                selected_columns.add(col_idx)

        # If no full columns are selected, get individual cell selections
        if not selected_columns:
            selected_indexes = selection_model.selectedIndexes()
            if not selected_indexes:
                return

            for idx in selected_indexes:
                selected_rows.add(idx.row())
                selected_columns.add(idx.column())
        else:
            # Full columns selected - include all rows
            selected_rows = set(range(self.table.rowCount()))

        if not selected_columns or not selected_rows:
            return

        # Sort for consistent output
        sorted_rows = sorted(selected_rows)
        sorted_columns = sorted(selected_columns)

        # Build clipboard text
        lines = []

        # Check if row headers are present
        has_row_headers = self.table.verticalHeader().isVisible()

        # Determine if we need CSV quoting (for comma separator)
        use_csv_quoting = (separator == ",")

        # Add headers if requested
        if include_headers:
            header_line = []

            # Add empty cell for row header column if row headers exist
            if has_row_headers:
                header_line.append("")

            # Add column headers
            for col in sorted_columns:
                header_item = self.table.horizontalHeaderItem(col)
                if header_item:
                    text = header_item.text()
                    header_line.append(self._quote_csv_field(text) if use_csv_quoting else text)
                else:
                    header_line.append("")
            lines.append(separator.join(header_line))

        # Add data rows
        for row in sorted_rows:
            row_values = []

            # Add row header if present
            if include_headers and has_row_headers:
                row_header_item = self.table.verticalHeaderItem(row)
                if row_header_item:
                    text = row_header_item.text()
                    row_values.append(self._quote_csv_field(text) if use_csv_quoting else text)
                else:
                    row_values.append(str(row))  # Use row number if no label

            # Add cell data
            for col in sorted_columns:
                item = self.table.item(row, col)
                if item:
                    text = item.text()
                    row_values.append(self._quote_csv_field(text) if use_csv_quoting else text)
                else:
                    row_values.append("")
            lines.append(separator.join(row_values))

        # Copy to clipboard
        clipboard_text = "\n".join(lines)
        clipboard = QApplication.clipboard()
        clipboard.setText(clipboard_text)

        # Show feedback if parent has status bar
        num_rows = len(sorted_rows)
        num_cols = len(sorted_columns)
        if hasattr(self, 'parent') and callable(self.parent):
            parent = self.parent()
            if parent and hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage(
                    f"Copied {num_rows} row(s) × {num_cols} column(s) to clipboard", 3000
                )

    def _quote_csv_field(self, text: str) -> str:
        """Quote a CSV field if it contains special characters.

        Args:
            text: The text to potentially quote

        Returns:
            Quoted text if necessary, otherwise original text
        """
        # Quote if field contains comma, newline, or double quote
        if ',' in text or '\n' in text or '\r' in text or '"' in text:
            # Escape double quotes by doubling them
            escaped = text.replace('"', '""')
            return f'"{escaped}"'
        return text
