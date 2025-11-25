from __future__ import annotations

import gzip
import os
import shutil
import tempfile
import h5py
import numpy as np
import csv
from qtpy.QtCore import QMimeData, Qt, QUrl, QFileInfo
from qtpy.QtGui import QIcon, QPainter, QPixmap, QColor, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QApplication, QStyle, QFileIconProvider
import pandas as pd


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
        self._icon_provider = QFileIconProvider()  # For system file icons
        self._filepath: str | None = None
        self._csv_filtered_indices = {}  # Dict mapping CSV group path to filtered row indices
        self._csv_visible_columns = {}  # Dict mapping CSV group path to list of visible column names
        self._csv_sort_specs = {}  # Dict mapping CSV group path to list of (column_name, ascending) tuples

    def flags(self, index):
        """Return item flags, enabling editing for groups and datasets."""
        if not index.isValid():
            return Qt.NoItemFlags

        # Only allow editing the name column
        if index.column() != self.COL_NAME:
            return super().flags(index)

        item = self.itemFromIndex(index)
        if item is None:
            return super().flags(index)

        kind = item.data(self.ROLE_KIND)

        # Allow editing for groups and datasets (but not root, attrs-folder, or attr)
        if kind in ("group", "dataset"):
            path = item.data(self.ROLE_PATH)
            # Don't allow editing the root group
            if kind == "group" and path == "/":
                return super().flags(index)
            return super().flags(index) | Qt.ItemIsEditable

        return super().flags(index)

    def setData(self, index, value, role=Qt.EditRole):
        """Handle data changes, including renaming groups and datasets in the HDF5 file."""
        if not index.isValid() or role != Qt.EditRole:
            return False

        if index.column() != self.COL_NAME:
            return False

        item = self.itemFromIndex(index)
        if item is None:
            return False

        kind = item.data(self.ROLE_KIND)
        old_path = item.data(self.ROLE_PATH)

        # Only handle groups and datasets
        if kind not in ("group", "dataset") or not old_path or old_path == "/":
            return False

        new_name = str(value).strip()
        if not new_name or new_name == item.text():
            return False

        # Validate new name (no slashes, not empty)
        if "/" in new_name:
            return False

        # Calculate new path
        parent_path = os.path.dirname(old_path)
        if parent_path == "":
            parent_path = "/"
        new_path = f"{parent_path}/{new_name}" if parent_path != "/" else f"/{new_name}"

        # Perform the rename in the HDF5 file
        if not self._filepath:
            return False

        try:
            with h5py.File(self._filepath, "r+") as h5:
                # Check if new path already exists
                if new_path in h5:
                    return False

                # Perform the move (rename)
                h5.move(old_path, new_path)

            # Update the item
            item.setText(new_name)
            item.setData(new_path, self.ROLE_PATH)

            # Update all descendant items' paths recursively
            self._update_descendant_paths(item, old_path, new_path)

            # Emit dataChanged signal
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True

        except Exception as e:  # noqa: BLE001
            print(f"Error renaming {old_path} to {new_path}: {e}")
            return False

    def _update_descendant_paths(self, parent_item, old_parent_path, new_parent_path):
        """Recursively update the ROLE_PATH data for all descendants after a rename.

        Args:
            parent_item: The item that was renamed
            old_parent_path: The old HDF5 path
            new_parent_path: The new HDF5 path
        """
        for row in range(parent_item.rowCount()):
            child_item = parent_item.child(row, self.COL_NAME)
            if child_item is None:
                continue

            old_child_path = child_item.data(self.ROLE_PATH)
            if old_child_path and old_child_path.startswith(old_parent_path):
                # Replace the old parent path prefix with the new one
                new_child_path = old_child_path.replace(old_parent_path, new_parent_path, 1)
                child_item.setData(new_child_path, self.ROLE_PATH)

                # Recursively update descendants
                self._update_descendant_paths(child_item, old_child_path, new_child_path)

    def _create_icon_with_indicator(self, base_icon: QIcon, has_attrs: bool) -> QIcon:
        """Create an icon with a red dot indicator if item has attributes.

        Args:
            base_icon: The base icon to use
            has_attrs: Whether to add the red dot indicator

        Returns:
            QIcon with red dot overlay if has_attrs is True, otherwise base_icon
        """
        if not has_attrs or base_icon.isNull():
            return base_icon

        # Get the base pixmap at a reasonable size
        size = 16
        pixmap = base_icon.pixmap(size, size)

        # Create a painter to draw on the pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw a red dot in the bottom-left corner
        dot_size = 5
        painter.setBrush(QColor(255, 0, 0))  # Red
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(1, size - dot_size - 1, dot_size, dot_size)

        painter.end()

        return QIcon(pixmap)

    def _create_image_thumbnail_icon(self, dataset: h5py.Dataset, has_attrs: bool) -> QIcon | None:
        """Create a thumbnail icon from an image dataset.

        Supports PNG, JPEG, GIF, BMP, and other formats supported by QPixmap.

        Args:
            dataset: HDF5 dataset containing image data
            has_attrs: Whether to add the red dot indicator

        Returns:
            QIcon with thumbnail, or None if data cannot be read
        """
        try:
            data = dataset[()]

            # Check if this is compressed binary data
            if "compressed" in dataset.attrs and dataset.attrs["compressed"] == "gzip":
                encoding = dataset.attrs.get("original_encoding", "utf-8")
                if isinstance(encoding, bytes):
                    encoding = encoding.decode("utf-8")
                if encoding == "binary" and isinstance(data, np.ndarray) and data.dtype == np.uint8:
                    # Decompress the binary data
                    compressed_bytes = data.tobytes()
                    img_bytes = gzip.decompress(compressed_bytes)
                elif isinstance(data, bytes):
                    img_bytes = data
                elif hasattr(data, "tobytes"):
                    img_bytes = data.tobytes()
                else:
                    return None
            elif isinstance(data, bytes):
                img_bytes = data
            elif hasattr(data, "tobytes"):
                img_bytes = data.tobytes()
            else:
                return None

            # Create pixmap from image data
            pixmap = QPixmap()
            if not pixmap.loadFromData(img_bytes):
                return None

            # Scale to thumbnail size (16x16 for tree view)
            thumbnail = pixmap.scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Add red dot indicator if has attributes
            if has_attrs:
                painter = QPainter(thumbnail)
                painter.setRenderHint(QPainter.Antialiasing)
                dot_size = 5
                painter.setBrush(QColor(255, 0, 0))  # Red
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(1, thumbnail.height() - dot_size - 1, dot_size, dot_size)
                painter.end()

            return QIcon(thumbnail)
        except Exception:  # noqa: BLE001
            return None

    def _get_system_icon_for_extension(self, filename: str, has_attrs: bool) -> QIcon | None:
        """Get the operating system's icon for a file extension.

        Args:
            filename: Name of the file (with extension)
            has_attrs: Whether to add red dot indicator

        Returns:
            QIcon with system icon (and optional red dot), or None if not available
        """
        try:

            ext = os.path.splitext(filename)[1].lower()
            if not ext:
                return None

            # Create a QFileInfo with just the filename (no actual file needed on modern Qt)
            file_info = QFileInfo(filename)
            icon = self._icon_provider.icon(file_info)

            if icon.isNull():
                return None

            # Add red dot indicator if item has attributes
            if has_attrs:
                return self._create_icon_with_indicator(icon, has_attrs)

            return icon
        except Exception:  # noqa: BLE001
            return None

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
        """Get the list of visible columns for a CSV group.

        Args:
            csv_group_path: HDF5 path to the CSV group

        Returns:
            List of visible column names, or None if not set
        """
        return self._csv_visible_columns.get(csv_group_path)

    def set_csv_sort_specs(self, csv_group_path: str, sort_specs: list[tuple[str, bool]] | None) -> None:
        """Set the sort specifications for a CSV group.

        Args:
            csv_group_path: HDF5 path to the CSV group
            sort_specs: List of (column_name, ascending) tuples, or None to clear
        """
        if sort_specs is None or not sort_specs:
            self._csv_sort_specs.pop(csv_group_path, None)
        else:
            self._csv_sort_specs[csv_group_path] = sort_specs

    def get_csv_sort_specs(self, csv_group_path: str) -> list[tuple[str, bool]] | None:
        """Get the sort specifications for a CSV group.

        Args:
            csv_group_path: HDF5 path to the CSV group

        Returns:
            List of (column_name, ascending) tuples, or None if not set
        """
        return self._csv_sort_specs.get(csv_group_path)

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
        self, group: h5py.Group, group_path: str, row_indices: np.ndarray | None = None, return_dataframe: bool = False, sort_specs: list[tuple[str, bool]] | None = None
    ) -> str | None:
        """Rebuild a CSV file from a CSV-derived group and return the temp file path or DataFrame.

        Uses 'column_names' attribute to determine column ordering if present.
        Falls back to sorted dataset names. Each dataset is expected to be 1-D (same length).

        Args:
            group: HDF5 group containing the CSV data
            group_path: Path to the group in the HDF5 file
            row_indices: Optional numpy array of row indices to export. If None, exports all rows.
            return_dataframe: If True, return a pandas DataFrame instead of writing a temp file.
            sort_specs: Optional list of (column_name, ascending) tuples for sorting. If None, checks stored sort specs.

        Returns:
            If return_dataframe is True, returns a pandas DataFrame or None on error.
            If return_dataframe is False, returns the path to a temporary CSV file or None on error.
        """
        # If sort_specs not provided, check if we have stored sort specs
        if sort_specs is None:
            sort_specs = self.get_csv_sort_specs(group_path)
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

            # Read columns - keep as numpy arrays to preserve types
            column_data: list = []  # List of numpy arrays or lists
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

                # Keep data in its original form (numpy array or scalar)
                if isinstance(data, np.ndarray):
                    # Decode byte strings if needed, but keep numeric types as-is
                    if data.dtype.kind == 'S' or data.dtype.kind == 'O':
                        # Byte strings or object arrays - decode to unicode
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
                        # Numeric or other types - keep as numpy array
                        column_data.append(data.ravel())
                    max_len = max(max_len, len(column_data[-1]))
                else:
                    column_data.append([str(data)])
                    max_len = max(max_len, 1)

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

            # Return DataFrame if requested
            if return_dataframe:
                # Build DataFrame from column data, preserving numpy array types
                data_dict = {}
                for idx, col_name in enumerate(col_names):
                    col_arr = column_data[idx]
                    # Apply row filtering
                    if isinstance(col_arr, np.ndarray):
                        # For numpy arrays, use fancy indexing
                        valid_indices = [i for i in export_indices if i < len(col_arr)]
                        if valid_indices:
                            filtered = col_arr[valid_indices]
                        else:
                            filtered = np.array([])
                        # Pad with NaN if needed (pandas will handle this correctly)
                        if len(valid_indices) < len(export_indices):
                            padding = [np.nan] * (len(export_indices) - len(valid_indices))
                            filtered = np.concatenate([filtered, padding])
                        data_dict[col_name] = filtered
                    else:
                        # For lists (string columns), use list comprehension
                        col_values = [col_arr[row_idx] if row_idx < len(col_arr) else ""
                                      for row_idx in export_indices]
                        data_dict[col_name] = col_values
                df = pd.DataFrame(data_dict)

                # Apply sorting if specified
                if sort_specs:
                    sort_columns = []
                    sort_orders = []
                    for col_name, ascending in sort_specs:
                        if col_name in df.columns:
                            sort_columns.append(col_name)
                            sort_orders.append(ascending)

                    if sort_columns:
                        try:
                            df = df.sort_values(by=sort_columns, ascending=sort_orders, na_position='last')
                        except Exception:
                            # If sorting fails, continue with unsorted data
                            pass

                return df

            # Write CSV (with sorting applied if needed)
            # Build DataFrame from column data, preserving numpy array types
            data_dict = {}
            for idx, col_name in enumerate(col_names):
                col_arr = column_data[idx]
                # Apply row filtering
                if isinstance(col_arr, np.ndarray):
                    # For numpy arrays, use fancy indexing
                    valid_indices = [i for i in export_indices if i < len(col_arr)]
                    if valid_indices:
                        filtered = col_arr[valid_indices]
                    else:
                        filtered = np.array([])
                    # Pad with NaN if needed (pandas will handle this correctly)
                    if len(valid_indices) < len(export_indices):
                        padding = [np.nan] * (len(export_indices) - len(valid_indices))
                        filtered = np.concatenate([filtered, padding])
                    data_dict[col_name] = filtered
                else:
                    # For lists (string columns), use list comprehension
                    col_values = [col_arr[row_idx] if row_idx < len(col_arr) else ""
                                  for row_idx in export_indices]
                    data_dict[col_name] = col_values
            df = pd.DataFrame(data_dict)

            # Apply sorting if specified
            if sort_specs:
                sort_columns = []
                sort_orders = []
                for col_name, ascending in sort_specs:
                    if col_name in df.columns:
                        sort_columns.append(col_name)
                        sort_orders.append(ascending)

                if sort_columns:
                    try:
                        df = df.sort_values(by=sort_columns, ascending=sort_orders, na_position='last')
                    except Exception:
                        # If sorting fails, continue with unsorted data
                        pass

            # Write sorted DataFrame to CSV
            df.to_csv(temp_path, index=False)
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
            has_attrs = len(group.attrs) > 0
            if csv_expanded:
                # Show folder icon when expanded
                base_icon = self._style.standardIcon(QStyle.SP_DirIcon)
                parent_item.setIcon(self._create_icon_with_indicator(base_icon, has_attrs))
            else:
                # Show table/dialog icon for collapsed CSV (makes them stand out)
                base_icon = self._style.standardIcon(QStyle.SP_FileDialogDetailedView)
                parent_item.setIcon(self._create_icon_with_indicator(base_icon, has_attrs))
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
                        has_attrs = len(obj.attrs) > 0
                        base_icon = self._style.standardIcon(QStyle.SP_DirIcon)
                        g_item.setIcon(self._create_icon_with_indicator(base_icon, has_attrs))
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
                        has_attrs = len(obj.attrs) > 0
                        icon_set = False

                        # Try to use image thumbnail for known image formats
                        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp', '.ico')
                        if name.lower().endswith(image_extensions):
                            img_icon = self._create_image_thumbnail_icon(obj, has_attrs)
                            if img_icon:
                                d_item.setIcon(img_icon)
                                icon_set = True

                        # If no image thumbnail, try system icon for the file extension
                        if not icon_set:
                            sys_icon = self._get_system_icon_for_extension(name, has_attrs)
                            if sys_icon:
                                d_item.setIcon(sys_icon)
                                icon_set = True

                        # Fallback to standard file icon if no system icon available
                        if not icon_set:
                            base_icon = self._style.standardIcon(QStyle.SP_FileIcon)
                            d_item.setIcon(self._create_icon_with_indicator(base_icon, has_attrs))
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
