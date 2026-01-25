"""
HDF5 storage backend implementation.

This module wraps h5py operations to conform to the StorageBackend interface.
"""

from __future__ import annotations

import json
import os
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .backend_interface import StorageBackend, StorageNode


class HDF5Backend(StorageBackend):
    """HDF5 implementation of the storage backend."""

    def __init__(self):
        """Initialize HDF5 backend."""
        self._file: h5py.File | None = None
        self._filepath: str | None = None
        self._mode: str = "r"

    def open(self, filepath: str, mode: str = "r") -> None:
        """Open an HDF5 file.

        Args:
            filepath: Path to the HDF5 file
            mode: Access mode ('r', 'r+', 'w', etc.)
        """
        if self._file is not None:
            self.close()
        self._file = h5py.File(filepath, mode)
        self._filepath = filepath
        self._mode = mode

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._filepath = None

    def create(self, filepath: str) -> None:
        """Create a new HDF5 file.

        Args:
            filepath: Path to the new HDF5 file
        """
        with h5py.File(filepath, "w"):
            pass
        self.open(filepath, "r+")

    def exists(self, path: str) -> bool:
        """Check if a node exists at the given path.

        Args:
            path: Path to check

        Returns:
            True if node exists, False otherwise
        """
        if self._file is None:
            return False
        return path in self._file

    def get_node(self, path: str) -> StorageNode:
        """Get information about a node.

        Args:
            path: Path to the node

        Returns:
            StorageNode with node information
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if path not in self._file:
            raise KeyError(f"Path not found: {path}")

        obj = self._file[path]
        name = os.path.basename(path) if path != "/" else "/"

        # Determine node type
        if isinstance(obj, h5py.Group):
            # Check if it's a CSV group
            if self.is_csv_group(path):
                node_type = "csv-group"
            else:
                node_type = "group"
            return StorageNode(path=path, name=name, node_type=node_type)
        elif isinstance(obj, h5py.Dataset):
            # Get dataset metadata
            shape = obj.shape
            dtype = str(obj.dtype)
            size = obj.size * obj.dtype.itemsize if hasattr(obj.dtype, "itemsize") else None
            compression = obj.compression
            compression_opts = obj.compression_opts
            chunks = obj.chunks

            return StorageNode(
                path=path,
                name=name,
                node_type="dataset",
                shape=shape,
                dtype=dtype,
                size=size,
                compression=compression,
                compression_opts=compression_opts,
                chunks=chunks,
            )
        else:
            return StorageNode(path=path, name=name, node_type="unknown")

    def list_children(self, path: str) -> list[StorageNode]:
        """List all children of a group node.

        Args:
            path: Path to the group node

        Returns:
            List of child StorageNode objects
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if path not in self._file:
            raise KeyError(f"Path not found: {path}")

        obj = self._file[path]
        if not isinstance(obj, h5py.Group):
            return []

        children = []
        for key in obj.keys():
            child_path = f"{path}/{key}" if path != "/" else f"/{key}"
            try:
                children.append(self.get_node(child_path))
            except Exception:
                # Skip nodes that can't be accessed
                continue

        return children

    def read_dataset(self, path: str) -> Any:
        """Read dataset data.

        Args:
            path: Path to the dataset

        Returns:
            Dataset data (numpy array, string, etc.)
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if path not in self._file:
            raise KeyError(f"Path not found: {path}")

        obj = self._file[path]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(f"Path is not a dataset: {path}")

        # Read the data
        data = obj[()]

        # Handle variable-length strings
        if h5py.check_vlen_dtype(obj.dtype):
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            elif isinstance(data, np.ndarray):
                # Array of strings
                return np.array([s.decode("utf-8", errors="replace") if isinstance(s, bytes) else s for s in data.flat]).reshape(data.shape)

        return data

    def write_dataset(
        self,
        path: str,
        data: Any,
        dtype: str | None = None,
        compression: str | None = None,
        compression_opts: Any = None,
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        """Write data to a dataset.

        Args:
            path: Path where dataset should be created
            data: Data to write
            dtype: Data type (optional)
            compression: Compression algorithm (optional)
            compression_opts: Compression options (optional)
            chunks: Chunk shape (optional)
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        # Delete existing dataset if it exists
        if path in self._file:
            del self._file[path]

        # Create parent groups if necessary
        parent_path = os.path.dirname(path)
        if parent_path and parent_path != "/" and parent_path not in self._file:
            self._file.create_group(parent_path)

        # Prepare kwargs
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if compression is not None:
            kwargs["compression"] = compression
        if compression_opts is not None:
            kwargs["compression_opts"] = compression_opts
        if chunks is not None:
            kwargs["chunks"] = chunks

        # Create dataset
        self._file.create_dataset(path, data=data, **kwargs)

    def create_group(self, path: str) -> None:
        """Create a group.

        Args:
            path: Path where group should be created
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if path in self._file:
            raise ValueError(f"Path already exists: {path}")

        # Create parent groups if necessary
        parent_path = os.path.dirname(path)
        if parent_path and parent_path != "/" and parent_path not in self._file:
            self._file.create_group(parent_path)

        self._file.create_group(path)

    def delete_node(self, path: str) -> None:
        """Delete a node (group or dataset).

        Args:
            path: Path to the node to delete
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if path not in self._file:
            raise KeyError(f"Path not found: {path}")

        del self._file[path]

    def rename_node(self, old_path: str, new_path: str) -> None:
        """Rename/move a node.

        Args:
            old_path: Current path
            new_path: New path
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if old_path not in self._file:
            raise KeyError(f"Path not found: {old_path}")

        if new_path in self._file:
            raise ValueError(f"Destination path already exists: {new_path}")

        self._file.move(old_path, new_path)

    def get_attributes(self, path: str) -> dict[str, Any]:
        """Get all attributes for a node.

        Args:
            path: Path to the node

        Returns:
            Dictionary of attribute key-value pairs
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if path not in self._file:
            raise KeyError(f"Path not found: {path}")

        obj = self._file[path]
        attrs = {}
        for key in obj.attrs.keys():
            value = obj.attrs[key]
            # Handle bytes
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            elif isinstance(value, np.ndarray) and value.dtype.kind == 'S':
                # Byte string array
                value = np.array([v.decode("utf-8", errors="replace") for v in value.flat]).reshape(value.shape)
            attrs[key] = value
        return attrs

    def set_attribute(self, path: str, key: str, value: Any) -> None:
        """Set an attribute on a node.

        Args:
            path: Path to the node
            key: Attribute key
            value: Attribute value
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if path not in self._file:
            raise KeyError(f"Path not found: {path}")

        obj = self._file[path]
        obj.attrs[key] = value

    def delete_attribute(self, path: str, key: str) -> None:
        """Delete an attribute from a node.

        Args:
            path: Path to the node
            key: Attribute key to delete
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if path not in self._file:
            raise KeyError(f"Path not found: {path}")

        obj = self._file[path]
        if key in obj.attrs:
            del obj.attrs[key]

    def copy_node(self, source_path: str, dest_path: str) -> None:
        """Copy a node to a new location.

        Args:
            source_path: Source path
            dest_path: Destination path
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if source_path not in self._file:
            raise KeyError(f"Source path not found: {source_path}")

        if dest_path in self._file:
            raise ValueError(f"Destination path already exists: {dest_path}")

        self._file.copy(source_path, dest_path)

    def get_filepath(self) -> str | None:
        """Get the current file path.

        Returns:
            File path or None if not opened
        """
        return self._filepath

    def is_csv_group(self, path: str) -> bool:
        """Check if a group is a CSV group.

        Args:
            path: Path to check

        Returns:
            True if path is a CSV group, False otherwise
        """
        if self._file is None:
            return False

        if path not in self._file:
            return False

        obj = self._file[path]
        if not isinstance(obj, h5py.Group):
            return False

        # Check for CSV marker attribute
        return obj.attrs.get("csv_group", False)

    def get_csv_dataframe(self, path: str) -> pd.DataFrame:
        """Get CSV data as a pandas DataFrame.

        Args:
            path: Path to CSV group

        Returns:
            pandas DataFrame with CSV data
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if not self.is_csv_group(path):
            raise ValueError(f"Path is not a CSV group: {path}")

        group = self._file[path]
        data = {}

        # Get column order from attribute
        column_order = group.attrs.get("column_order", None)
        if column_order is not None:
            if isinstance(column_order, bytes):
                column_order = json.loads(column_order.decode("utf-8"))
            elif isinstance(column_order, str):
                column_order = json.loads(column_order)
        else:
            column_order = sorted(group.keys())

        # Read each column dataset
        for col_name in column_order:
            if col_name in group:
                data[col_name] = group[col_name][()]

        return pd.DataFrame(data)

    def get_csv_filtered_indices(self, path: str) -> list[int] | None:
        """Get filtered row indices for a CSV group.

        Args:
            path: Path to CSV group

        Returns:
            List of filtered row indices, or None if no filter
        """
        if self._file is None:
            return None

        if not self.is_csv_group(path):
            return None

        group = self._file[path]
        filter_data = group.attrs.get("_filter_indices", None)
        if filter_data is None:
            return None

        if isinstance(filter_data, bytes):
            filter_json = filter_data.decode("utf-8")
        else:
            filter_json = str(filter_data)

        return json.loads(filter_json)

    def set_csv_filtered_indices(self, path: str, indices: list[int] | None) -> None:
        """Set filtered row indices for a CSV group.

        Args:
            path: Path to CSV group
            indices: List of filtered row indices, or None to clear filter
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if not self.is_csv_group(path):
            raise ValueError(f"Path is not a CSV group: {path}")

        group = self._file[path]
        if indices is None:
            if "_filter_indices" in group.attrs:
                del group.attrs["_filter_indices"]
        else:
            group.attrs["_filter_indices"] = json.dumps(indices)

    def get_csv_visible_columns(self, path: str) -> list[str] | None:
        """Get list of visible column names for a CSV group.

        Args:
            path: Path to CSV group

        Returns:
            List of visible column names, or None for all columns
        """
        if self._file is None:
            return None

        if not self.is_csv_group(path):
            return None

        group = self._file[path]
        visible_data = group.attrs.get("_visible_columns", None)
        if visible_data is None:
            return None

        if isinstance(visible_data, bytes):
            visible_json = visible_data.decode("utf-8")
        else:
            visible_json = str(visible_data)

        return json.loads(visible_json)

    def set_csv_visible_columns(self, path: str, columns: list[str] | None) -> None:
        """Set list of visible column names for a CSV group.

        Args:
            path: Path to CSV group
            columns: List of visible column names, or None for all columns
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if not self.is_csv_group(path):
            raise ValueError(f"Path is not a CSV group: {path}")

        group = self._file[path]
        if columns is None:
            if "_visible_columns" in group.attrs:
                del group.attrs["_visible_columns"]
        else:
            group.attrs["_visible_columns"] = json.dumps(columns)

    def get_csv_sort_spec(self, path: str) -> list[tuple[str, bool]] | None:
        """Get sort specification for a CSV group.

        Args:
            path: Path to CSV group

        Returns:
            List of (column_name, ascending) tuples, or None
        """
        if self._file is None:
            return None

        if not self.is_csv_group(path):
            return None

        group = self._file[path]
        sort_data = group.attrs.get("_sort_spec", None)
        if sort_data is None:
            return None

        if isinstance(sort_data, bytes):
            sort_json = sort_data.decode("utf-8")
        else:
            sort_json = str(sort_data)

        sort_list = json.loads(sort_json)
        return [(item[0], item[1]) for item in sort_list]

    def set_csv_sort_spec(self, path: str, sort_spec: list[tuple[str, bool]] | None) -> None:
        """Set sort specification for a CSV group.

        Args:
            path: Path to CSV group
            sort_spec: List of (column_name, ascending) tuples, or None
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        if not self.is_csv_group(path):
            raise ValueError(f"Path is not a CSV group: {path}")

        group = self._file[path]
        if sort_spec is None:
            if "_sort_spec" in group.attrs:
                del group.attrs["_sort_spec"]
        else:
            group.attrs["_sort_spec"] = json.dumps(sort_spec)

    def create_csv_group(
        self,
        path: str,
        dataframe: pd.DataFrame,
        compression: str = "gzip",
        compression_opts: int = 9,
    ) -> None:
        """Create a CSV group from a pandas DataFrame.

        Args:
            path: Path where CSV group should be created
            dataframe: pandas DataFrame with CSV data
            compression: Compression algorithm
            compression_opts: Compression level
        """
        if self._file is None:
            raise RuntimeError("No file is currently open")

        # Create group
        if path in self._file:
            del self._file[path]

        group = self._file.create_group(path)

        # Mark as CSV group
        group.attrs["csv_group"] = True

        # Store column order
        group.attrs["column_order"] = json.dumps(list(dataframe.columns))

        # Create dataset for each column
        for col_name in dataframe.columns:
            col_data = dataframe[col_name].values
            group.create_dataset(
                col_name,
                data=col_data,
                compression=compression,
                compression_opts=compression_opts,
            )

    def export_csv_group(self, path: str, output_file: str) -> None:
        """Export a CSV group to a CSV file.

        Args:
            path: Path to CSV group
            output_file: Output CSV file path
        """
        df = self.get_csv_dataframe(path)
        df.to_csv(output_file, index=False)

    def import_file(
        self,
        disk_path: str,
        storage_path: str,
        compression: str = "gzip",
        compression_level: int = 9,
    ) -> None:
        """Import a file from disk into HDF5 storage.

        For CSV files, creates a CSV group with individual datasets.
        For other files, stores as compressed text or binary data.

        Args:
            disk_path: Path to file on disk
            storage_path: Path in HDF5 where file should be imported
            compression: Compression algorithm to use
            compression_level: Compression level (1-9)
        """
        import gzip

        if self._file is None:
            raise RuntimeError("No file is open")

        # Check if storage_path already exists
        if storage_path in self._file:
            raise FileExistsError(storage_path)

        # Special handling for CSV files
        if disk_path.lower().endswith(".csv"):
            # Read CSV as DataFrame
            df = pd.read_csv(disk_path)
            # Remove .csv extension from storage path
            if storage_path.lower().endswith(".csv"):
                storage_path = storage_path[:-4]
            self.create_csv_group(storage_path, df, compression, compression_level)
            return

        # Ensure parent groups exist
        parent = os.path.dirname(storage_path).replace("\\", "/")
        if parent and parent != "/":
            if parent not in self._file:
                self._file.require_group(parent)

        # Try to read as text, fallback to binary
        try:
            with open(disk_path, "r", encoding="utf-8") as f:
                data = f.read()
            # Compress text data
            compressed = gzip.compress(data.encode("utf-8"), compresslevel=compression_level)
            ds = self._file.create_dataset(
                storage_path, data=np.frombuffer(compressed, dtype="uint8")
            )
            ds.attrs["compressed"] = compression
            ds.attrs["original_encoding"] = "utf-8"
        except (UnicodeDecodeError, Exception):
            # Read as binary and compress
            with open(disk_path, "rb") as f:
                data = f.read()
            compressed = gzip.compress(data, compresslevel=compression_level)
            ds = self._file.create_dataset(
                storage_path, data=np.frombuffer(compressed, dtype="uint8")
            )
            ds.attrs["compressed"] = compression
            ds.attrs["original_encoding"] = "binary"

    def import_folder(
        self,
        disk_folder: str,
        storage_path: str,
        compression: str = "gzip",
        compression_level: int = 9,
        excluded_dirs: set[str] | None = None,
        excluded_files: set[str] | None = None,
    ) -> tuple[int, list[str]]:
        """Import a folder recursively from disk into HDF5 storage.

        Args:
            disk_folder: Path to folder on disk
            storage_path: Path in HDF5 where folder should be imported
            compression: Compression algorithm to use
            compression_level: Compression level (1-9)
            excluded_dirs: Set of directory names to exclude
            excluded_files: Set of file names to exclude

        Returns:
            Tuple of (files_imported, error_messages)
        """
        import posixpath

        if self._file is None:
            raise RuntimeError("No file is open")

        if excluded_dirs is None:
            excluded_dirs = set()
        if excluded_files is None:
            excluded_files = set()

        imported = 0
        errors = []

        base_name = os.path.basename(os.path.normpath(disk_folder))
        if storage_path == "/":
            root_storage_path = "/" + base_name
        else:
            root_storage_path = posixpath.join(storage_path, base_name)

        # Walk the directory tree
        for dirpath, dirnames, filenames in os.walk(disk_folder):
            # Filter out excluded directories
            dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

            # Calculate relative path
            rel = os.path.relpath(dirpath, disk_folder)
            rel = "." if rel == "." else rel.replace("\\", "/")

            # Calculate current storage path
            current_storage_path = (
                root_storage_path if rel == "." else posixpath.join(root_storage_path, rel)
            )

            # Ensure the group exists
            if current_storage_path not in self._file:
                self._file.require_group(current_storage_path)

            # Import each file
            for filename in filenames:
                if filename in excluded_files:
                    continue

                file_on_disk = os.path.join(dirpath, filename)
                file_storage_path = posixpath.join(current_storage_path, filename)

                try:
                    self.import_file(file_on_disk, file_storage_path, compression, compression_level)
                    imported += 1
                except FileExistsError:
                    # File already exists - this should be handled by caller
                    raise
                except Exception as exc:
                    errors.append(f"{file_storage_path}: {exc}")

        return imported, errors
