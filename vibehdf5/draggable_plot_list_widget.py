"""
QListWidget that supports drag-and-drop to export plots to filesystem.
"""

import os
import tempfile
import traceback

from qtpy.QtWidgets import QListWidget, QAbstractItemView, QMessageBox
from qtpy.QtGui import QDrag
from qtpy.QtCore import QMimeData, QUrl, Qt


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
                QMessageBox.warning(
                    self,
                    "Export Failed",
                    f"Failed to export plot for drag-and-drop.\n\nError: {error_msg}",
                )
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.warning(self, "Export Error", f"Error exporting plot: {e}\n\n{tb}")
