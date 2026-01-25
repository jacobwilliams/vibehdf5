"""
SQLite storage backend implementation using SQLAlchemy.

This module implements the StorageBackend interface using SQLite
to store hierarchical data structures similar to HDF5.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .backend_interface import StorageBackend, StorageNode
from .sqlite_models import (
    Attribute,
    Base,
    CSVColumn,
    CSVMetadata,
    Dataset,
    Node,
)


class SQLiteBackend(StorageBackend):
    """SQLite implementation of the storage backend using SQLAlchemy."""

    def __init__(self):
        """Initialize SQLite backend."""
        self.engine = None
        self.session_factory = None
        self._filepath: str | None = None
        self._session: Session | None = None

    def open(self, filepath: str, mode: str = "r") -> None:
        """Open a SQLite database file.

        Args:
            filepath: Path to the SQLite database file
            mode: Access mode ('r', 'r+', 'w', etc.)
        """
        if self.engine is not None:
            self.close()

        # Create SQLAlchemy engine
        db_url = f"sqlite:///{filepath}"
        self.engine = create_engine(db_url)
        self.session_factory = sessionmaker(bind=self.engine)
        self._session = self.session_factory()
        self._filepath = filepath

        # If file doesn't exist or mode is 'w', create tables
        if mode == "w" or not os.path.exists(filepath):
            Base.metadata.create_all(self.engine)
            # Create root node if it doesn't exist
            self._ensure_root_node()

    def close(self) -> None:
        """Close the SQLite database."""
        if self._session is not None:
            self._session.close()
            self._session = None
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            self.session_factory = None
        self._filepath = None

    def create(self, filepath: str) -> None:
        """Create a new SQLite database file.

        Args:
            filepath: Path to the new SQLite database file
        """
        # Remove existing file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)

        self.open(filepath, "w")

    def _ensure_root_node(self) -> None:
        """Ensure root node exists in the database."""
        if self._session is None:
            return

        root = self._session.execute(
            select(Node).where(Node.path == "/")
        ).scalar_one_or_none()

        if root is None:
            root = Node(path="/", name="/", node_type="group", parent_path=None)
            self._session.add(root)
            self._session.commit()

    def exists(self, path: str) -> bool:
        """Check if a node exists at the given path.

        Args:
            path: Path to check

        Returns:
            True if node exists, False otherwise
        """
        if self._session is None:
            return False

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        return node is not None

    def get_node(self, path: str) -> StorageNode:
        """Get information about a node.

        Args:
            path: Path to the node

        Returns:
            StorageNode with node information
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        # Build StorageNode based on node type
        if node.node_type == "dataset" and node.dataset:
            ds = node.dataset
            shape = tuple(json.loads(ds.shape)) if ds.shape else None
            chunks = tuple(json.loads(ds.chunks)) if ds.chunks else None
            comp_opts = json.loads(ds.compression_opts) if ds.compression_opts else None

            return StorageNode(
                path=node.path,
                name=node.name,
                node_type=node.node_type,
                shape=shape,
                dtype=ds.dtype,
                size=len(ds.data) if ds.data else 0,
                compression=ds.compression,
                compression_opts=comp_opts,
                chunks=chunks,
            )
        else:
            return StorageNode(
                path=node.path,
                name=node.name,
                node_type=node.node_type,
            )

    def list_children(self, path: str) -> list[StorageNode]:
        """List all children of a group node.

        Args:
            path: Path to the group node

        Returns:
            List of child StorageNode objects
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        # Get all nodes whose parent_path matches
        children_nodes = self._session.execute(
            select(Node).where(Node.parent_path == path)
        ).scalars().all()

        children = []
        for node in children_nodes:
            try:
                children.append(self.get_node(node.path))
            except Exception:
                continue

        return children

    def read_dataset(self, path: str) -> Any:
        """Read dataset data.

        Args:
            path: Path to the dataset

        Returns:
            Dataset data (numpy array, string, etc.)
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        if node.node_type != "dataset" or node.dataset is None:
            raise ValueError(f"Path is not a dataset: {path}")

        ds = node.dataset
        data = ds.data

        # Decompress if needed
        if ds.compression == "gzip":
            data = gzip.decompress(data)

        # Deserialize based on dtype
        if ds.dtype and ds.dtype.startswith("str") or ds.dtype == "object":
            # String data
            return data.decode("utf-8", errors="replace")
        else:
            # Numerical data - deserialize numpy array
            arr = np.frombuffer(data, dtype=ds.dtype if ds.dtype else "uint8")
            if ds.shape:
                shape = tuple(json.loads(ds.shape))
                arr = arr.reshape(shape)
            return arr

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
        if self._session is None:
            raise RuntimeError("No database is currently open")

        # Delete existing node if it exists
        if self.exists(path):
            self.delete_node(path)

        # Ensure parent exists
        parent_path = os.path.dirname(path) if path != "/" else None
        if parent_path and not self.exists(parent_path):
            self.create_group(parent_path)

        # Determine parent_path for database
        if parent_path == "":
            parent_path = "/"

        # Convert data to bytes
        if isinstance(data, str):
            # String data
            data_bytes = data.encode("utf-8")
            if dtype is None:
                dtype = "str"
        elif isinstance(data, (int, float)):
            # Scalar
            arr = np.array([data])
            data_bytes = arr.tobytes()
            if dtype is None:
                dtype = str(arr.dtype)
            shape_json = json.dumps([1])
        elif isinstance(data, np.ndarray):
            # Numpy array
            data_bytes = data.tobytes()
            if dtype is None:
                dtype = str(data.dtype)
            shape_json = json.dumps(list(data.shape))
        else:
            # Try to convert to numpy array
            arr = np.array(data)
            data_bytes = arr.tobytes()
            if dtype is None:
                dtype = str(arr.dtype)
            shape_json = json.dumps(list(arr.shape))

        # Apply compression if specified
        if compression == "gzip":
            comp_level = compression_opts if compression_opts is not None else 9
            data_bytes = gzip.compress(data_bytes, compresslevel=comp_level)

        # Create node
        name = os.path.basename(path)
        node = Node(
            path=path,
            name=name,
            node_type="dataset",
            parent_path=parent_path,
        )
        self._session.add(node)
        self._session.flush()  # Get node.id

        # Create dataset
        dataset = Dataset(
            node_id=node.id,
            dtype=dtype,
            shape=shape_json if isinstance(data, np.ndarray) or isinstance(data, (list, tuple)) else None,
            data=data_bytes,
            compression=compression,
            compression_opts=json.dumps(compression_opts) if compression_opts is not None else None,
            chunks=json.dumps(list(chunks)) if chunks else None,
        )
        self._session.add(dataset)
        self._session.commit()

    def create_group(self, path: str) -> None:
        """Create a group.

        Args:
            path: Path where group should be created
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        if self.exists(path):
            raise ValueError(f"Path already exists: {path}")

        # Ensure parent exists
        parent_path = os.path.dirname(path)
        if parent_path and parent_path != "/" and not self.exists(parent_path):
            self.create_group(parent_path)

        if parent_path == "":
            parent_path = "/"

        # Create node
        name = os.path.basename(path)
        node = Node(
            path=path,
            name=name,
            node_type="group",
            parent_path=parent_path,
        )
        self._session.add(node)
        self._session.commit()

    def delete_node(self, path: str) -> None:
        """Delete a node (group or dataset).

        Args:
            path: Path to the node to delete
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        # Get the node
        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        # Delete all descendants recursively
        self._delete_descendants(path)

        # Delete the node itself (cascading will delete related records)
        self._session.delete(node)
        self._session.commit()

    def _delete_descendants(self, path: str) -> None:
        """Recursively delete all descendants of a node."""
        if self._session is None:
            return

        # Get all direct children
        children = self._session.execute(
            select(Node).where(Node.parent_path == path)
        ).scalars().all()

        for child in children:
            # Recursively delete descendants
            self._delete_descendants(child.path)
            # Delete the child
            self._session.delete(child)

    def rename_node(self, old_path: str, new_path: str) -> None:
        """Rename/move a node.

        Args:
            old_path: Current path
            new_path: New path
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        # Get the node
        node = self._session.execute(
            select(Node).where(Node.path == old_path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {old_path}")

        if self.exists(new_path):
            raise ValueError(f"Destination path already exists: {new_path}")

        # Update the node
        old_name = node.name
        new_name = os.path.basename(new_path)
        new_parent = os.path.dirname(new_path)
        if new_parent == "":
            new_parent = "/"

        node.path = new_path
        node.name = new_name
        node.parent_path = new_parent

        # Update all descendants' paths
        self._update_descendant_paths(old_path, new_path)

        self._session.commit()

    def _update_descendant_paths(self, old_parent_path: str, new_parent_path: str) -> None:
        """Recursively update paths of all descendants after a rename."""
        if self._session is None:
            return

        # Get all nodes whose path starts with old_parent_path
        all_nodes = self._session.execute(select(Node)).scalars().all()

        for node in all_nodes:
            if node.path.startswith(old_parent_path + "/"):
                # Update the path
                new_path = node.path.replace(old_parent_path, new_parent_path, 1)
                node.path = new_path

                # Update parent_path if needed
                if node.parent_path == old_parent_path:
                    node.parent_path = new_parent_path
                elif node.parent_path and node.parent_path.startswith(old_parent_path + "/"):
                    node.parent_path = node.parent_path.replace(old_parent_path, new_parent_path, 1)

    def get_attributes(self, path: str) -> dict[str, Any]:
        """Get all attributes for a node.

        Args:
            path: Path to the node

        Returns:
            Dictionary of attribute key-value pairs
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        attrs = {}
        for attr in node.attributes:
            # Deserialize value based on type
            if attr.value_type == "str":
                attrs[attr.key] = attr.value
            elif attr.value_type == "int":
                attrs[attr.key] = int(attr.value)
            elif attr.value_type == "float":
                attrs[attr.key] = float(attr.value)
            elif attr.value_type == "bool":
                attrs[attr.key] = attr.value.lower() == "true"
            elif attr.value_type in ("list", "array", "dict"):
                attrs[attr.key] = json.loads(attr.value)
            else:
                attrs[attr.key] = attr.value

        return attrs

    def set_attribute(self, path: str, key: str, value: Any) -> None:
        """Set an attribute on a node.

        Args:
            path: Path to the node
            key: Attribute key
            value: Attribute value
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        # Determine value type and serialize
        if isinstance(value, bool):
            value_type = "bool"
            value_str = str(value)
        elif isinstance(value, int):
            value_type = "int"
            value_str = str(value)
        elif isinstance(value, float):
            value_type = "float"
            value_str = str(value)
        elif isinstance(value, str):
            value_type = "str"
            value_str = value
        elif isinstance(value, (list, tuple)):
            value_type = "list"
            value_str = json.dumps(value)
        elif isinstance(value, dict):
            value_type = "dict"
            value_str = json.dumps(value)
        elif isinstance(value, np.ndarray):
            value_type = "array"
            value_str = json.dumps(value.tolist())
        else:
            value_type = "str"
            value_str = str(value)

        # Check if attribute already exists
        attr = self._session.execute(
            select(Attribute).where(
                Attribute.node_id == node.id,
                Attribute.key == key
            )
        ).scalar_one_or_none()

        if attr is None:
            # Create new attribute
            attr = Attribute(
                node_id=node.id,
                key=key,
                value_type=value_type,
                value=value_str,
            )
            self._session.add(attr)
        else:
            # Update existing attribute
            attr.value_type = value_type
            attr.value = value_str

        self._session.commit()

    def delete_attribute(self, path: str, key: str) -> None:
        """Delete an attribute from a node.

        Args:
            path: Path to the node
            key: Attribute key to delete
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        # Find and delete the attribute
        attr = self._session.execute(
            select(Attribute).where(
                Attribute.node_id == node.id,
                Attribute.key == key
            )
        ).scalar_one_or_none()

        if attr is not None:
            self._session.delete(attr)
            self._session.commit()

    def copy_node(self, source_path: str, dest_path: str) -> None:
        """Copy a node to a new location.

        Args:
            source_path: Source path
            dest_path: Destination path
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        source_node = self.get_node(source_path)

        if source_node.node_type == "dataset":
            # Copy dataset
            data = self.read_dataset(source_path)
            self.write_dataset(
                dest_path,
                data,
                dtype=source_node.dtype,
                compression=source_node.compression,
                compression_opts=source_node.compression_opts,
                chunks=source_node.chunks,
            )
            # Copy attributes
            attrs = self.get_attributes(source_path)
            for key, value in attrs.items():
                self.set_attribute(dest_path, key, value)
        else:
            # Copy group
            self.create_group(dest_path)
            # Copy attributes
            attrs = self.get_attributes(source_path)
            for key, value in attrs.items():
                self.set_attribute(dest_path, key, value)
            # Recursively copy children
            children = self.list_children(source_path)
            for child in children:
                child_dest = f"{dest_path}/{child.name}"
                self.copy_node(child.path, child_dest)

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
        if not self.exists(path):
            return False

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        return node is not None and node.node_type == "csv-group"

    def get_csv_dataframe(self, path: str) -> pd.DataFrame:
        """Get CSV data as a pandas DataFrame.

        Args:
            path: Path to CSV group

        Returns:
            pandas DataFrame with CSV data
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        if not self.is_csv_group(path):
            raise ValueError(f"Path is not a CSV group: {path}")

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        # Get all CSV columns ordered by column_index
        columns = self._session.execute(
            select(CSVColumn).where(CSVColumn.node_id == node.id).order_by(CSVColumn.column_index)
        ).scalars().all()

        data = {}
        for col in columns:
            # Decompress and deserialize column data
            col_data = col.data
            if col_data:
                # Handle different data types
                if col.dtype and col.dtype.startswith("object"):
                    # String/object data - deserialize from JSON
                    data[col.column_name] = json.loads(col_data.decode("utf-8"))
                else:
                    # Numerical data - deserialize from bytes
                    arr = np.frombuffer(col_data, dtype=col.dtype if col.dtype else "float64")
                    data[col.column_name] = arr

        return pd.DataFrame(data)

    def get_csv_filtered_indices(self, path: str) -> list[int] | None:
        """Get filtered row indices for a CSV group.

        Args:
            path: Path to CSV group

        Returns:
            List of filtered row indices, or None if no filter
        """
        if self._session is None:
            return None

        if not self.is_csv_group(path):
            return None

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None or node.csv_metadata is None:
            return None

        if node.csv_metadata.filtered_indices:
            return json.loads(node.csv_metadata.filtered_indices)
        return None

    def set_csv_filtered_indices(self, path: str, indices: list[int] | None) -> None:
        """Set filtered row indices for a CSV group.

        Args:
            path: Path to CSV group
            indices: List of filtered row indices, or None to clear filter
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        if not self.is_csv_group(path):
            raise ValueError(f"Path is not a CSV group: {path}")

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        # Get or create CSV metadata
        if node.csv_metadata is None:
            metadata = CSVMetadata(node_id=node.id)
            self._session.add(metadata)
            self._session.flush()
        else:
            metadata = node.csv_metadata

        metadata.filtered_indices = json.dumps(indices) if indices is not None else None
        self._session.commit()

    def get_csv_visible_columns(self, path: str) -> list[str] | None:
        """Get list of visible column names for a CSV group.

        Args:
            path: Path to CSV group

        Returns:
            List of visible column names, or None for all columns
        """
        if self._session is None:
            return None

        if not self.is_csv_group(path):
            return None

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            return None

        # Get visible columns from CSVColumn table
        columns = self._session.execute(
            select(CSVColumn).where(
                CSVColumn.node_id == node.id,
                CSVColumn.visible == True  # noqa: E712
            ).order_by(CSVColumn.column_index)
        ).scalars().all()

        if not columns:
            return None

        return [col.column_name for col in columns]

    def set_csv_visible_columns(self, path: str, columns: list[str] | None) -> None:
        """Set list of visible column names for a CSV group.

        Args:
            path: Path to CSV group
            columns: List of visible column names, or None for all columns
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        if not self.is_csv_group(path):
            raise ValueError(f"Path is not a CSV group: {path}")

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        # Get all CSV columns
        all_columns = self._session.execute(
            select(CSVColumn).where(CSVColumn.node_id == node.id)
        ).scalars().all()

        if columns is None:
            # Make all columns visible
            for col in all_columns:
                col.visible = True
        else:
            # Set visibility based on list
            for col in all_columns:
                col.visible = col.column_name in columns

        self._session.commit()

    def get_csv_sort_spec(self, path: str) -> list[tuple[str, bool]] | None:
        """Get sort specification for a CSV group.

        Args:
            path: Path to CSV group

        Returns:
            List of (column_name, ascending) tuples, or None
        """
        if self._session is None:
            return None

        if not self.is_csv_group(path):
            return None

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None or node.csv_metadata is None:
            return None

        if node.csv_metadata.sort_spec:
            sort_list = json.loads(node.csv_metadata.sort_spec)
            return [(item[0], item[1]) for item in sort_list]
        return None

    def set_csv_sort_spec(self, path: str, sort_spec: list[tuple[str, bool]] | None) -> None:
        """Set sort specification for a CSV group.

        Args:
            path: Path to CSV group
            sort_spec: List of (column_name, ascending) tuples, or None
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        if not self.is_csv_group(path):
            raise ValueError(f"Path is not a CSV group: {path}")

        node = self._session.execute(
            select(Node).where(Node.path == path)
        ).scalar_one_or_none()

        if node is None:
            raise KeyError(f"Path not found: {path}")

        # Get or create CSV metadata
        if node.csv_metadata is None:
            metadata = CSVMetadata(node_id=node.id)
            self._session.add(metadata)
            self._session.flush()
        else:
            metadata = node.csv_metadata

        metadata.sort_spec = json.dumps(sort_spec) if sort_spec is not None else None
        self._session.commit()

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
            compression: Compression algorithm (currently ignored for SQLite)
            compression_opts: Compression level (currently ignored for SQLite)
        """
        if self._session is None:
            raise RuntimeError("No database is currently open")

        # Delete existing node if it exists
        if self.exists(path):
            self.delete_node(path)

        # Ensure parent exists
        parent_path = os.path.dirname(path)
        if parent_path and parent_path != "/" and not self.exists(parent_path):
            self.create_group(parent_path)

        if parent_path == "":
            parent_path = "/"

        # Create CSV group node
        name = os.path.basename(path)
        node = Node(
            path=path,
            name=name,
            node_type="csv-group",
            parent_path=parent_path,
        )
        self._session.add(node)
        self._session.flush()  # Get node.id

        # Mark as CSV group with attribute
        self.set_attribute(path, "csv_group", True)

        # Create CSV columns
        for idx, col_name in enumerate(dataframe.columns):
            col_data = dataframe[col_name].values
            dtype_str = str(col_data.dtype)

            # Handle different data types
            if dtype_str.startswith("object") or dtype_str.startswith("<U") or dtype_str.startswith("|S"):
                # String/object data - serialize as JSON
                col_bytes = json.dumps(col_data.tolist()).encode("utf-8")
            else:
                # Numerical data - serialize as bytes
                col_bytes = col_data.tobytes()

            csv_col = CSVColumn(
                node_id=node.id,
                column_name=col_name,
                column_index=idx,
                dtype=dtype_str,
                data=col_bytes,
                visible=True,
            )
            self._session.add(csv_col)

        self._session.commit()

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
        """Import a file from disk into SQLite storage.

        For CSV files, creates a CSV group with individual datasets.
        For other files, stores as compressed text or binary data.

        Args:
            disk_path: Path to file on disk
            storage_path: Path in storage where file should be imported
            compression: Compression algorithm to use
            compression_level: Compression level (1-9)
        """
        import gzip

        if self._session is None:
            raise RuntimeError("No database is open")

        # Check if storage_path already exists
        if self.exists(storage_path):
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
        if parent and parent != "/" and not self.exists(parent):
            self.create_group(parent)

        # Try to read as text, fallback to binary
        original_encoding = "utf-8"
        try:
            with open(disk_path, "r", encoding="utf-8") as f:
                data = f.read()
            compressed = gzip.compress(data.encode("utf-8"), compresslevel=compression_level)
        except (UnicodeDecodeError, Exception):
            # Read as binary and compress
            with open(disk_path, "rb") as f:
                data = f.read()
            compressed = gzip.compress(data, compresslevel=compression_level)
            original_encoding = "binary"

        # Store as dataset
        self.write_dataset(storage_path, np.frombuffer(compressed, dtype="uint8"))

        # Set compression attributes
        self.set_attribute(storage_path, "compressed", compression)
        self.set_attribute(storage_path, "original_encoding", original_encoding)

    def import_folder(
        self,
        disk_folder: str,
        storage_path: str,
        compression: str = "gzip",
        compression_level: int = 9,
        excluded_dirs: set[str] | None = None,
        excluded_files: set[str] | None = None,
    ) -> tuple[int, list[str]]:
        """Import a folder recursively from disk into SQLite storage.

        Args:
            disk_folder: Path to folder on disk
            storage_path: Path in storage where folder should be imported
            compression: Compression algorithm to use
            compression_level: Compression level (1-9)
            excluded_dirs: Set of directory names to exclude
            excluded_files: Set of file names to exclude

        Returns:
            Tuple of (files_imported, error_messages)
        """
        import posixpath

        if self._session is None:
            raise RuntimeError("No database is open")

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
            if not self.exists(current_storage_path):
                self.create_group(current_storage_path)

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
