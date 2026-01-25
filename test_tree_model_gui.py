"""
Test script to verify tree model can load both HDF5 and SQLite files.

This creates sample files and tests that the tree model can load them.
"""

import os
import tempfile

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

from vibehdf5.backend_factory import create_backend
from vibehdf5.hdf5_tree_model import HDF5TreeModel


def create_test_files():
    """Create test HDF5 and SQLite files with sample data."""
    temp_dir = tempfile.gettempdir()

    # Create HDF5 file
    hdf5_path = os.path.join(temp_dir, "test_gui.h5")
    backend = create_backend(hdf5_path, "hdf5")
    backend.create(hdf5_path)

    # Add some data
    backend.create_group("/data")
    backend.write_dataset("/data/array", np.array([1, 2, 3, 4, 5]))
    backend.write_dataset("/data/text", "Hello from HDF5")
    backend.set_attribute("/data", "description", "Test data group")

    # Add CSV group
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95.5, 87.2, 91.8]
    })
    backend.create_csv_group("/csv_data", df)

    backend.close()
    print(f"✓ Created HDF5 test file: {hdf5_path}")

    # Create SQLite file
    sqlite_path = os.path.join(temp_dir, "test_gui.db")
    backend = create_backend(sqlite_path, "sqlite")
    backend.create(sqlite_path)

    # Add some data
    backend.create_group("/data")
    backend.write_dataset("/data/array", np.array([1, 2, 3, 4, 5]))
    backend.write_dataset("/data/text", "Hello from SQLite")
    backend.set_attribute("/data", "description", "Test data group")

    # Add CSV group
    backend.create_csv_group("/csv_data", df)

    backend.close()
    print(f"✓ Created SQLite test file: {sqlite_path}")

    return hdf5_path, sqlite_path


def test_tree_model():
    """Test that the tree model can load both file types."""
    # Create QApplication (required for Qt widgets)
    app = QApplication.instance()
    if not app:
        app = QApplication([])

    hdf5_path, sqlite_path = create_test_files()

    try:
        # Test HDF5 file
        print("\nTesting HDF5 file loading...")
        model = HDF5TreeModel()
        model.load_file(hdf5_path)

        # Verify root item exists
        root = model.invisibleRootItem()
        assert root.rowCount() > 0, "Root should have children"

        # Get first child (the file root)
        file_item = root.child(0, 0)
        assert file_item is not None, "File item should exist"
        print(f"  ✓ File root: {file_item.text()}")

        # Verify file item has children (groups/datasets)
        assert file_item.rowCount() > 0, "File should have child items"
        print(f"  ✓ Has {file_item.rowCount()} child items")

        print("✅ HDF5 file loaded successfully!")

        # Test SQLite file
        print("\nTesting SQLite file loading...")
        model2 = HDF5TreeModel()
        model2.load_file(sqlite_path)

        # Verify root item exists
        root2 = model2.invisibleRootItem()
        assert root2.rowCount() > 0, "Root should have children"

        # Get first child (the file root)
        file_item2 = root2.child(0, 0)
        assert file_item2 is not None, "File item should exist"
        print(f"  ✓ File root: {file_item2.text()}")

        # Verify file item has children (groups/datasets)
        assert file_item2.rowCount() > 0, "File should have child items"
        print(f"  ✓ Has {file_item2.rowCount()} child items")

        print("✅ SQLite file loaded successfully!")

        print("\n" + "=" * 60)
        print("✅ ALL TREE MODEL TESTS PASSED!")
        print("=" * 60)

    finally:
        # Cleanup
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
            print(f"\n✓ Cleaned up: {hdf5_path}")
        if os.path.exists(sqlite_path):
            os.remove(sqlite_path)
            print(f"✓ Cleaned up: {sqlite_path}")


if __name__ == "__main__":
    test_tree_model()
