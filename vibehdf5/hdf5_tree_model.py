from __future__ import annotations

import gzip
import os
import shutil
import tempfile
import h5py
import numpy as np
import csv
from qtpy.QtCore import QMimeData, Qt, QUrl
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QApplication, QStyle


class HDF5TreeModel(QStandardItemModel):
    """A simple tree model to display HDF5 file structure.

    Columns:
    - Name: group/dataset/attribute name
    - Info: type, shape, dtype (for datasets), attribute value preview
    """

    COL_NAME = 0
    COL_INFO = 1

    ROLE_PATH = Qt.UserRole + 1
    ROLE_KIND = Qt.UserRole + 2  # 'file', 'group', 'dataset', 'attr', 'attrs-folder'
    ROLE_ATTR_KEY = Qt.UserRole + 3
    ROLE_CSV_EXPANDED = Qt.UserRole + 4  # True if CSV group's internal structure is shown

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Name", "Info"])
        self._style = QApplication.instance().style() if QApplication.instance() else None
        self._filepath: str | None = None
        self._csv_filtered_indices = {}  # Dict mapping CSV group path to filtered row indices
        self._csv_visible_columns = {}  # Dict mapping CSV group path to list of visible column names

    def load_file(self, filepath: str) -> None:
        """Load the HDF5 file and populate the model."""
        self.clear()
        self.setHorizontalHeaderLabels(["Name", "Info"])  # reset headers after clear()
        self._filepath = filepath

        root_name = filepath.split("/")[-1]
        root_item = QStandardItem(root_name)
        info_item = QStandardItem("HDF5 file")

        if self._style:
            root_item.setIcon(self._style.standardIcon(QStyle.SP_DriveHDIcon))

        self.invisibleRootItem().appendRow([root_item, info_item])
        root_item.setData("/", self.ROLE_PATH)
        root_item.setData("file", self.ROLE_KIND)

        with h5py.File(filepath, "r") as h5:
            self._add_group(h5, root_item)

    @property
    def filepath(self) -> str | None:
        return self._filepath

    def set_csv_filtered_indices(self, csv_group_path: str, indices: np.ndarray | None) -> None:
        """Set the filtered row indices for a CSV group.

        Args:
            csv_group_path: HDF5 path to the CSV group
            indices: numpy array of row indices to export, or None to export all rows
        """
        if indices is None:
            self._csv_filtered_indices.pop(csv_group_path, None)
        else:
            self._csv_filtered_indices[csv_group_path] = indices

    def set_csv_visible_columns(self, csv_group_path: str, visible_columns: list[str] | None) -> None:
        """Set the visible columns for a CSV group export.

        Args:
            csv_group_path: HDF5 path to the CSV group
            visible_columns: list of column names to export, or None to export all columns
        """
        if visible_columns is None or not visible_columns:
            self._csv_visible_columns.pop(csv_group_path, None)
        else:
            self._csv_visible_columns[csv_group_path] = visible_columns

    def get_csv_visible_columns(self, csv_group_path: str) -> list[str] | None:
        """Get the visible columns for a CSV group.

        Args:
            csv_group_path: HDF5 path to the CSV group

        Returns:
            List of visible column names, or None if all columns should be shown
        """
        return self._csv_visible_columns.get(csv_group_path)

    def get_csv_filtered_indices(self, csv_group_path: str) -> np.ndarray | None:
        """Get the filtered row indices for a CSV group.

        Args:
            csv_group_path: HDF5 path to the CSV group

        Returns:
            numpy array of row indices, or None if no filtering active
        """
        return self._csv_filtered_indices.get(csv_group_path)

    def supportedDragActions(self):
        """Enable copy action for drag-and-drop."""
        return Qt.CopyAction

    def mimeTypes(self):
        """Specify that we provide file URLs for drag-and-drop."""
        return ["text/uri-list"]

    def mimeData(self, indexes):
        """Create mime data containing a temporary file/folder with the dataset/group content."""
        if not indexes:
            return None

        # Get the first index (should be column 0)
        index = indexes[0]
        if index.column() != self.COL_NAME:
            # Find the corresponding column 0 index
            index = self.index(index.row(), self.COL_NAME, index.parent())

        item = self.itemFromIndex(index)
        if item is None:
            return None

        kind = item.data(self.ROLE_KIND)
        path = item.data(self.ROLE_PATH)

        # Only allow dragging datasets and groups (not attributes or file root)
        if kind not in ("dataset", "group"):
            return None

        # Don't allow dragging the root group
        if kind == "group" and path == "/":
            return None

        # Store internal HDF5 path for internal moves
        mime = QMimeData()
        mime.setData("application/x-hdf5-path", path.encode("utf-8"))

        if not self._filepath:
            return None

        try:
            if kind == "dataset":
                # Extract single dataset to a file
                with h5py.File(self._filepath, "r") as h5:
                    ds = h5[path]
                    if not isinstance(ds, h5py.Dataset):
                        return None

                    # Determine filename from the dataset path
                    dataset_name = os.path.basename(path)
                    if not dataset_name:
                        dataset_name = "dataset"

                    # Create a temporary file
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, dataset_name)

                    self._save_dataset_to_file(ds, temp_path)

                    # Add file URL to mime data
                    url = QUrl.fromLocalFile(temp_path)
                    mime.setUrls([url])
                    return mime

            elif kind == "group":
                # If this group represents a CSV (has source_type=='csv'),
                # reconstruct a CSV file instead of a folder tree.
                with h5py.File(self._filepath, "r") as h5:
                    group = h5[path]
                    if not isinstance(group, h5py.Group):
                        return None

                    is_csv = (
                        "source_type" in group.attrs
                        and str(group.attrs["source_type"]).lower() == "csv"
                    )

                    if is_csv:
                        # Get filtered indices for this CSV group (if any)
                        filtered_indices = self.get_csv_filtered_indices(path)
                        csv_path = self._reconstruct_csv_tempfile(group, path, filtered_indices)
                        if not csv_path:
                            return None
                        url = QUrl.fromLocalFile(csv_path)
                        mime.setUrls([url])
                        return mime

                    # Fallback: extract group as folder hierarchy
                    group_name = os.path.basename(path) or "group"
                    temp_dir = tempfile.gettempdir()
                    temp_folder = os.path.join(temp_dir, group_name)
                    if os.path.exists(temp_folder):
                        shutil.rmtree(temp_folder)
                    os.makedirs(temp_folder, exist_ok=True)
                    self._extract_group_to_folder(group, temp_folder)
                    url = QUrl.fromLocalFile(temp_folder)
                    mime.setUrls([url])
                    return mime

        except Exception:  # noqa: BLE001
            return None

    def _save_dataset_to_file(self, ds, file_path):
        """Save a single dataset to a file.

        Automatically decompresses gzip-compressed text datasets.
        """

        # Read dataset content
        data = ds[()]

        # Check if this is a gzip-compressed text dataset
        try:
            if "compressed" in ds.attrs and ds.attrs["compressed"] == "gzip":
                if isinstance(data, np.ndarray) and data.dtype == np.uint8:
                    compressed_bytes = data.tobytes()
                    decompressed = gzip.decompress(compressed_bytes)
                    encoding = ds.attrs.get("original_encoding", "utf-8")
                    if isinstance(encoding, bytes):
                        encoding = encoding.decode("utf-8")
                    # Check if this is binary data
                    if encoding == "binary":
                        # Write decompressed binary data
                        with open(file_path, "wb") as f:
                            f.write(decompressed)
                        return
                    # Otherwise it's text
                    text = decompressed.decode(encoding)
                    with open(file_path, "w", encoding=encoding) as f:
                        f.write(text)
                    return
        except Exception:  # noqa: BLE001
            pass

        # Try to save based on data type
        if isinstance(data, np.ndarray) and data.dtype == np.uint8:
            # Binary data (like images)
            with open(file_path, "wb") as f:
                f.write(data.tobytes())
        elif isinstance(data, (bytes, bytearray)):
            # Raw bytes
            with open(file_path, "wb") as f:
                f.write(data)
        elif isinstance(data, str):
            # String data
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(data)
        else:
            # Try string representation
            vld = h5py.check_string_dtype(ds.dtype)
            if vld is not None:
                # Variable-length string
                as_str = ds.asstr()[()]
                if isinstance(as_str, np.ndarray):
                    text = "\n".join(map(str, as_str.ravel().tolist()))
                else:
                    text = str(as_str)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                # Fallback: convert to text
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(data))

    def _extract_group_to_folder(self, group, folder_path):
        """Recursively extract a group and its contents to a folder."""

        # Iterate through all items in the group
        for name, obj in group.items():
            if isinstance(obj, h5py.Dataset):
                # Save dataset as a file
                file_path = os.path.join(folder_path, name)
                self._save_dataset_to_file(obj, file_path)
            elif isinstance(obj, h5py.Group):
                # Create subfolder and recurse
                subfolder_path = os.path.join(folder_path, name)
                os.makedirs(subfolder_path, exist_ok=True)
                self._extract_group_to_folder(obj, subfolder_path)

    @staticmethod
    def _sanitize_hdf5_name(name: str) -> str:
        try:
            s = (name or "").strip()
            s = s.replace("/", "_")
            return s or "unnamed"
        except Exception:  # noqa: BLE001
            return "unnamed"

    def _reconstruct_csv_tempfile(
        self, group: h5py.Group, group_path: str, row_indices: np.ndarray | None = None
    ) -> str | None:
        """Rebuild a CSV file from a CSV-derived group and return the temp file path.

        Uses 'column_names' attribute to determine column ordering if present.
        Falls back to sorted dataset names. Each dataset is expected to be 1-D (same length).

        Args:
            group: HDF5 group containing the CSV data
            group_path: Path to the group in the HDF5 file
            row_indices: Optional numpy array of row indices to export. If None, exports all rows.
        """
        try:
            # Determine filename
            source_file = group.attrs.get("source_file")
            if isinstance(source_file, (bytes, bytearray)):
                try:
                    source_file = source_file.decode("utf-8")
                except Exception:  # noqa: BLE001
                    source_file = None
            if isinstance(source_file, str) and source_file.lower().endswith(".csv"):
                fname = source_file
            else:
                # Use group name + .csv
                fname = (os.path.basename(group_path) or "group") + ".csv"

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, fname)

            # Column names
            raw_cols = group.attrs.get("column_names")
            if raw_cols is not None:
                # h5py may give numpy array; convert to list of str
                try:
                    col_names = [str(c) for c in list(raw_cols)]
                except Exception:  # noqa: BLE001
                    col_names = []
            else:
                col_names = []

            if not col_names:
                # Fallback to dataset keys
                col_names = [name for name in group.keys() if isinstance(group[name], h5py.Dataset)]
                col_names.sort()

            # Filter to only visible columns if specified
            visible_columns = self._csv_visible_columns.get(group_path)
            if visible_columns:
                # Keep only columns that are in the visible list (preserve order)
                col_names = [col for col in col_names if col in visible_columns]

            # If present, use explicit mapping of column -> dataset name
            col_ds_names = None
            raw_map = group.attrs.get("column_dataset_names")
            if raw_map is not None:
                try:
                    col_ds_names = [str(c) for c in list(raw_map)]
                    if len(col_ds_names) != len(col_names):
                        col_ds_names = None
                except Exception:  # noqa: BLE001
                    col_ds_names = None

            # Read columns
            column_data: list[list[str]] = []
            max_len = 0
            for idx, col in enumerate(col_names):
                if col_ds_names is not None:
                    key = col_ds_names[idx]
                else:
                    # Try sanitized version of the column
                    key = self._sanitize_hdf5_name(col)
                    if key not in group and col in group:
                        key = col
                if key not in group:
                    column_data.append([])
                    continue
                obj = group[key]
                if not isinstance(obj, h5py.Dataset):
                    column_data.append([])
                    continue
                data = obj[()]
                # Normalize to list of strings
                if isinstance(data, np.ndarray):
                    # For byte strings or object, coerce each element
                    entries = []
                    for v in data.ravel().tolist():
                        if isinstance(v, bytes):
                            try:
                                entries.append(v.decode("utf-8"))
                            except Exception:  # noqa: BLE001
                                entries.append(v.decode("utf-8", "replace"))
                        else:
                            entries.append(str(v))
                    column_data.append(entries)
                else:
                    column_data.append([str(data)])
                max_len = max(max_len, len(column_data[-1]))

            # Align columns (pad shorter columns with empty strings)
            for col_list in column_data:
                if len(col_list) < max_len:
                    col_list.extend([""] * (max_len - len(col_list)))

            # Determine which rows to export
            if row_indices is not None:
                # Export only filtered rows
                export_indices = row_indices
            else:
                # Export all rows
                export_indices = np.arange(max_len)

            # Write CSV
            with open(temp_path, "w", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)
                writer.writerow(col_names)
                for row_idx in export_indices:
                    if row_idx < max_len:
                        row = [column_data[c][row_idx] for c in range(len(col_names))]
                        writer.writerow(row)
            return temp_path
        except Exception:  # noqa: BLE001
            return None

    # Internal helpers
    def _add_group(self, group: h5py.Group, parent_item: QStandardItem) -> None:
        """Recursively add a group and its children to the model."""
        # Check if this is a CSV-derived group
        is_csv_group = False
        try:
            if "source_type" in group.attrs and group.attrs["source_type"] == "csv":
                is_csv_group = True
        except Exception:  # noqa: BLE001
            pass

        # Attributes (if any) - only show if not CSV or if CSV is expanded
        csv_expanded = parent_item.data(self.ROLE_CSV_EXPANDED) or False

        # Set icon for CSV groups based on expansion state
        if is_csv_group and self._style:
            if csv_expanded:
                # Show folder icon when expanded
                parent_item.setIcon(self._style.standardIcon(QStyle.SP_DirIcon))
            else:
                # Show table/dialog icon for collapsed CSV (makes them stand out)
                parent_item.setIcon(self._style.standardIcon(QStyle.SP_FileDialogDetailedView))
        if len(group.attrs) and (not is_csv_group or csv_expanded):
            attrs_item = QStandardItem("Attributes")
            attrs_info = QStandardItem(f"{len(group.attrs)} item(s)")
            if self._style:
                attrs_item.setIcon(self._style.standardIcon(QStyle.SP_DirIcon))
            parent_item.appendRow([attrs_item, attrs_info])
            attrs_item.setData(group.name, self.ROLE_PATH)
            attrs_item.setData("attrs-folder", self.ROLE_KIND)
            for key, val in group.attrs.items():
                name_item = QStandardItem(str(key))
                value_preview = _value_preview(val)
                info_item = QStandardItem(f"attr = {value_preview}")
                if self._style:
                    name_item.setIcon(self._style.standardIcon(QStyle.SP_MessageBoxInformation))
                attrs_item.appendRow([name_item, info_item])
                name_item.setData(group.name, self.ROLE_PATH)
                name_item.setData("attr", self.ROLE_KIND)
                name_item.setData(str(key), self.ROLE_ATTR_KEY)

        # Child groups and datasets - only show if not CSV or if CSV is expanded
        if not is_csv_group or csv_expanded:
            for name, obj in group.items():
                if isinstance(obj, h5py.Group):
                    g_item = QStandardItem(name)
                    g_info = QStandardItem("group")
                    if self._style:
                        g_item.setIcon(self._style.standardIcon(QStyle.SP_DirIcon))
                    parent_item.appendRow([g_item, g_info])
                    g_item.setData(obj.name, self.ROLE_PATH)
                    g_item.setData("group", self.ROLE_KIND)
                    self._add_group(obj, g_item)
                elif isinstance(obj, h5py.Dataset):
                    d_item = QStandardItem(name)
                    shape = obj.shape
                    dtype = obj.dtype
                    space = f"{shape}" if shape is not None else "(scalar)"
                    d_info = QStandardItem(f"dataset | shape={space} | dtype={dtype}")
                    if self._style:
                        d_item.setIcon(self._style.standardIcon(QStyle.SP_FileIcon))
                    parent_item.appendRow([d_item, d_info])
                    d_item.setData(obj.name, self.ROLE_PATH)
                    d_item.setData("dataset", self.ROLE_KIND)
                else:  # pragma: no cover - unknown kinds
                    unk_item = QStandardItem(name)
                    unk_info = QStandardItem(type(obj).__name__)
                    parent_item.appendRow([unk_item, unk_info])

    def toggle_csv_group_expansion(self, item: QStandardItem) -> None:
        """Toggle the expansion of a CSV group's internal structure and reload."""
        if item is None:
            return

        kind = item.data(self.ROLE_KIND)
        path = item.data(self.ROLE_PATH)

        if kind != "group" or not path or not self._filepath:
            return

        # Check if this is a CSV group
        try:
            with h5py.File(self._filepath, "r") as h5:
                grp = h5[path]
                if not isinstance(grp, h5py.Group):
                    return
                if "source_type" not in grp.attrs or grp.attrs["source_type"] != "csv":
                    return

                # Toggle expansion state
                is_expanded = item.data(self.ROLE_CSV_EXPANDED) or False
                item.setData(not is_expanded, self.ROLE_CSV_EXPANDED)

                # Remove all children
                item.removeRows(0, item.rowCount())

                # Re-add group content with new expansion state
                self._add_group(grp, item)
        except Exception:  # noqa: BLE001
            pass


def _value_preview(val, max_len: int = 80) -> str:
    """Create a compact one-line preview for attribute values."""
    try:
        text = repr(val)
    except Exception:
        text = str(val)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text
