"""
Abstract backend interface for storage systems.

This module defines the abstract interface that all storage backends
(HDF5, SQLite, etc.) must implement to work with the viewer GUI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator


class StorageNode:
    """Represents a node in the hierarchical storage structure."""

    def __init__(
        self,
        path: str,
        name: str,
        node_type: str,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
        size: int | None = None,
        compression: str | None = None,
        compression_opts: Any = None,
        chunks: tuple[int, ...] | None = None,
    ):
        """Initialize a storage node.

        Args:
            path: Full path to the node (e.g., "/group1/dataset1")
            name: Node name (last component of path)
            node_type: Type of node ('file', 'group', 'dataset', 'attr', 'attrs-folder', 'csv-group')
            shape: Shape of dataset (for datasets)
            dtype: Data type (for datasets)
            size: Size in bytes (for datasets)
            compression: Compression algorithm (for datasets)
            compression_opts: Compression options (for datasets)
            chunks: Chunk shape (for datasets)
        """
        self.path = path
        self.name = name
        self.node_type = node_type
        self.shape = shape
        self.dtype = dtype
        self.size = size
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunks = chunks


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def open(self, filepath: str, mode: str = "r") -> None:
        """Open a storage file.

        Args:
            filepath: Path to the storage file
            mode: Access mode ('r', 'r+', 'w', etc.)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the storage file."""
        pass

    @abstractmethod
    def create(self, filepath: str) -> None:
        """Create a new storage file.

        Args:
            filepath: Path to the new storage file
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a node exists at the given path.

        Args:
            path: Path to check

        Returns:
            True if node exists, False otherwise
        """
        pass

    @abstractmethod
    def get_node(self, path: str) -> StorageNode:
        """Get information about a node.

        Args:
            path: Path to the node

        Returns:
            StorageNode with node information
        """
        pass

    @abstractmethod
    def list_children(self, path: str) -> list[StorageNode]:
        """List all children of a group node.

        Args:
            path: Path to the group node

        Returns:
            List of child StorageNode objects
        """
        pass

    @abstractmethod
    def read_dataset(self, path: str) -> Any:
        """Read dataset data.

        Args:
            path: Path to the dataset

        Returns:
            Dataset data (numpy array, string, etc.)
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def create_group(self, path: str) -> None:
        """Create a group.

        Args:
            path: Path where group should be created
        """
        pass

    @abstractmethod
    def delete_node(self, path: str) -> None:
        """Delete a node (group or dataset).

        Args:
            path: Path to the node to delete
        """
        pass

    @abstractmethod
    def rename_node(self, old_path: str, new_path: str) -> None:
        """Rename/move a node.

        Args:
            old_path: Current path
            new_path: New path
        """
        pass

    @abstractmethod
    def get_attributes(self, path: str) -> dict[str, Any]:
        """Get all attributes for a node.

        Args:
            path: Path to the node

        Returns:
            Dictionary of attribute key-value pairs
        """
        pass

    @abstractmethod
    def set_attribute(self, path: str, key: str, value: Any) -> None:
        """Set an attribute on a node.

        Args:
            path: Path to the node
            key: Attribute key
            value: Attribute value
        """
        pass

    @abstractmethod
    def delete_attribute(self, path: str, key: str) -> None:
        """Delete an attribute from a node.

        Args:
            path: Path to the node
            key: Attribute key to delete
        """
        pass

    @abstractmethod
    def copy_node(self, source_path: str, dest_path: str) -> None:
        """Copy a node to a new location.

        Args:
            source_path: Source path
            dest_path: Destination path
        """
        pass

    @abstractmethod
    def get_filepath(self) -> str | None:
        """Get the current file path.

        Returns:
            File path or None if not opened
        """
        pass

    @abstractmethod
    def is_csv_group(self, path: str) -> bool:
        """Check if a group is a CSV group.

        Args:
            path: Path to check

        Returns:
            True if path is a CSV group, False otherwise
        """
        pass

    @abstractmethod
    def get_csv_dataframe(self, path: str) -> Any:
        """Get CSV data as a pandas DataFrame.

        Args:
            path: Path to CSV group

        Returns:
            pandas DataFrame with CSV data
        """
        pass

    @abstractmethod
    def get_csv_filtered_indices(self, path: str) -> list[int] | None:
        """Get filtered row indices for a CSV group.

        Args:
            path: Path to CSV group

        Returns:
            List of filtered row indices, or None if no filter
        """
        pass

    @abstractmethod
    def set_csv_filtered_indices(self, path: str, indices: list[int] | None) -> None:
        """Set filtered row indices for a CSV group.

        Args:
            path: Path to CSV group
            indices: List of filtered row indices, or None to clear filter
        """
        pass

    @abstractmethod
    def get_csv_visible_columns(self, path: str) -> list[str] | None:
        """Get list of visible column names for a CSV group.

        Args:
            path: Path to CSV group

        Returns:
            List of visible column names, or None for all columns
        """
        pass

    @abstractmethod
    def set_csv_visible_columns(self, path: str, columns: list[str] | None) -> None:
        """Set list of visible column names for a CSV group.

        Args:
            path: Path to CSV group
            columns: List of visible column names, or None for all columns
        """
        pass

    @abstractmethod
    def get_csv_sort_spec(self, path: str) -> list[tuple[str, bool]] | None:
        """Get sort specification for a CSV group.

        Args:
            path: Path to CSV group

        Returns:
            List of (column_name, ascending) tuples, or None
        """
        pass

    @abstractmethod
    def set_csv_sort_spec(self, path: str, sort_spec: list[tuple[str, bool]] | None) -> None:
        """Set sort specification for a CSV group.

        Args:
            path: Path to CSV group
            sort_spec: List of (column_name, ascending) tuples, or None
        """
        pass

    @abstractmethod
    def create_csv_group(
        self,
        path: str,
        dataframe: Any,
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
        pass

    @abstractmethod
    def export_csv_group(self, path: str, output_file: str) -> None:
        """Export a CSV group to a CSV file.

        Args:
            path: Path to CSV group
            output_file: Output CSV file path
        """
        pass

    @abstractmethod
    def import_file(
        self,
        disk_path: str,
        storage_path: str,
        compression: str = "gzip",
        compression_level: int = 9,
    ) -> None:
        """Import a file from disk into storage.

        For CSV files, creates a CSV group with individual datasets.
        For other files, stores as compressed text or binary data.

        Args:
            disk_path: Path to file on disk
            storage_path: Path in storage where file should be imported
            compression: Compression algorithm to use
            compression_level: Compression level (1-9)
        """
        pass

    @abstractmethod
    def import_folder(
        self,
        disk_folder: str,
        storage_path: str,
        compression: str = "gzip",
        compression_level: int = 9,
        excluded_dirs: set[str] | None = None,
        excluded_files: set[str] | None = None,
    ) -> tuple[int, list[str]]:
        """Import a folder recursively from disk into storage.

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
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
