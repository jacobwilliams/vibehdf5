"""
TreeView that accepts external file/folder drops and forwards to the viewer.
"""

import os

from qtpy.QtWidgets import QMessageBox, QTreeView


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
        """HDF5Viewer instance to forward drag-and-drop operations to"""
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
