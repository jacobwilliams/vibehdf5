"""
Factory for creating storage backend instances.

This module provides a factory function to create the appropriate
storage backend based on file extension.
"""

from __future__ import annotations

import os
from typing import Literal

from .backend_interface import StorageBackend
from .hdf5_backend import HDF5Backend
from .sqlite_backend import SQLiteBackend


def create_backend(filepath: str | None = None, backend_type: Literal["hdf5", "sqlite", "auto"] = "auto") -> StorageBackend:
    """Create a storage backend instance.

    Args:
        filepath: Path to the storage file (used to infer backend type if backend_type is "auto")
        backend_type: Type of backend to create ("hdf5", "sqlite", or "auto")

    Returns:
        StorageBackend instance

    Raises:
        ValueError: If backend_type is invalid or cannot be inferred
    """
    if backend_type == "auto":
        if filepath is None:
            raise ValueError("filepath must be provided when backend_type is 'auto'")

        # Infer backend from file extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".h5", ".hdf5", ".hdf"):
            backend_type = "hdf5"
        elif ext in (".db", ".sqlite", ".sqlite3"):
            backend_type = "sqlite"
        else:
            # Default to HDF5 for unknown extensions
            backend_type = "hdf5"

    if backend_type == "hdf5":
        return HDF5Backend()
    elif backend_type == "sqlite":
        return SQLiteBackend()
    else:
        raise ValueError(f"Invalid backend_type: {backend_type}")


def get_supported_extensions() -> dict[str, list[str]]:
    """Get supported file extensions for each backend.

    Returns:
        Dictionary mapping backend type to list of extensions
    """
    return {
        "hdf5": [".h5", ".hdf5", ".hdf"],
        "sqlite": [".db", ".sqlite", ".sqlite3"],
    }


def get_all_extensions() -> list[str]:
    """Get all supported file extensions.

    Returns:
        List of all supported file extensions
    """
    extensions = []
    for exts in get_supported_extensions().values():
        extensions.extend(exts)
    return extensions


def infer_backend_type(filepath: str) -> Literal["hdf5", "sqlite"]:
    """Infer the backend type from a file path.

    Args:
        filepath: Path to the storage file

    Returns:
        Backend type ("hdf5" or "sqlite")
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in get_supported_extensions()["sqlite"]:
        return "sqlite"
    else:
        # Default to HDF5
        return "hdf5"
