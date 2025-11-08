from __future__ import annotations

import h5py
import numpy as np
import tempfile
import os
from pathlib import Path
import shutil
from qtpy.QtCore import Qt, QUrl, QMimeData
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QStyle, QApplication


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Name", "Info"])
        self._style = QApplication.instance().style() if QApplication.instance() else None
        self._filepath: str | None = None

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

                    # Create mime data with file URL
                    mime = QMimeData()
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
                        'source_type' in group.attrs
                        and str(group.attrs['source_type']).lower() == 'csv'
                    )

                    if is_csv:
                        csv_path = self._reconstruct_csv_tempfile(group, path)
                        if not csv_path:
                            return None
                        mime = QMimeData()
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
                    mime = QMimeData()
                    url = QUrl.fromLocalFile(temp_folder)
                    mime.setUrls([url])
                    return mime

        except Exception:
            return None

    def _save_dataset_to_file(self, ds, file_path):
        """Save a single dataset to a file."""

        # Read dataset content
        data = ds[()]

        # Try to save based on data type
        if isinstance(data, np.ndarray) and data.dtype == np.uint8:
            # Binary data (like images)
            with open(file_path, 'wb') as f:
                f.write(data.tobytes())
        elif isinstance(data, (bytes, bytearray)):
            # Raw bytes
            with open(file_path, 'wb') as f:
                f.write(data)
        elif isinstance(data, str):
            # String data
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            # Try string representation
            vld = h5py.check_string_dtype(ds.dtype)
            if vld is not None:
                # Variable-length string
                as_str = ds.asstr()[()]
                if isinstance(as_str, np.ndarray):
                    text = '\n'.join(map(str, as_str.ravel().tolist()))
                else:
                    text = str(as_str)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            else:
                # Fallback: convert to text
                with open(file_path, 'w', encoding='utf-8') as f:
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

    def _reconstruct_csv_tempfile(self, group: h5py.Group, group_path: str) -> str | None:
        """Rebuild a CSV file from a CSV-derived group and return the temp file path.

        Uses 'column_names' attribute to determine column ordering if present.
        Falls back to sorted dataset names. Each dataset is expected to be 1-D (same length).
        """
        try:
            # Determine filename
            source_file = group.attrs.get('source_file')
            if isinstance(source_file, (bytes, bytearray)):
                try:
                    source_file = source_file.decode('utf-8')
                except Exception:
                    source_file = None
            if isinstance(source_file, str) and source_file.lower().endswith('.csv'):
                fname = source_file
            else:
                # Use group name + .csv
                fname = (os.path.basename(group_path) or 'group') + '.csv'

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, fname)

            # Column names
            raw_cols = group.attrs.get('column_names')
            if raw_cols is not None:
                # h5py may give numpy array; convert to list of str
                try:
                    col_names = [str(c) for c in list(raw_cols)]
                except Exception:
                    col_names = []
            else:
                col_names = []

            if not col_names:
                # Fallback to dataset keys
                col_names = [name for name in group.keys() if isinstance(group[name], h5py.Dataset)]
                col_names.sort()

            # Read columns
            column_data: list[list[str]] = []
            max_len = 0
            for col in col_names:
                ds_name = col.strip()
                if ds_name not in group:
                    column_data.append([])
                    continue
                obj = group[ds_name]
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
                                entries.append(v.decode('utf-8'))
                            except Exception:
                                entries.append(v.decode('utf-8', 'replace'))
                        else:
                            entries.append(str(v))
                    column_data.append(entries)
                else:
                    column_data.append([str(data)])
                max_len = max(max_len, len(column_data[-1]))

            # Align columns (pad shorter columns with empty strings)
            for col_list in column_data:
                if len(col_list) < max_len:
                    col_list.extend([''] * (max_len - len(col_list)))

            # Write CSV
            import csv
            with open(temp_path, 'w', newline='', encoding='utf-8') as fout:
                writer = csv.writer(fout)
                writer.writerow(col_names)
                for row_idx in range(max_len):
                    row = [column_data[c][row_idx] for c in range(len(col_names))]
                    writer.writerow(row)
            return temp_path
        except Exception:
            return None

    # Internal helpers
    def _add_group(self, group: h5py.Group, parent_item: QStandardItem) -> None:
        """Recursively add a group and its children to the model."""
        # Attributes (if any)
        if len(group.attrs):
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

        # Child groups and datasets
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


def _value_preview(val, max_len: int = 80) -> str:
    """Create a compact one-line preview for attribute values."""
    try:
        text = repr(val)
    except Exception:
        text = str(val)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text
