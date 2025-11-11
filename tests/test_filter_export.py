#!/usr/bin/env python3
"""Test script to verify CSV filtering and export functionality."""

import h5py
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vibehdf5.hdf5_tree_model import HDF5TreeModel


def create_test_file():
    """Create a test HDF5 file with CSV data."""
    temp_path = os.path.join(tempfile.gettempdir(), "test_filter_export.h5")

    with h5py.File(temp_path, "w") as f:
        # Create a CSV group
        csv_group = f.create_group("test_data")
        csv_group.attrs["source_type"] = "csv"
        csv_group.attrs["source_file"] = "test_data.csv"
        csv_group.attrs["column_names"] = ["Name", "Age", "Score"]

        # Add data
        names = ["Alice", "Bob", "Charlie", "David", "Eve"]
        ages = [25, 30, 35, 40, 45]
        scores = [85.5, 90.0, 75.5, 88.0, 92.5]

        csv_group.create_dataset("Name", data=np.array(names, dtype='S'))
        csv_group.create_dataset("Age", data=np.array(ages))
        csv_group.create_dataset("Score", data=np.array(scores))

    print(f"Created test file: {temp_path}")
    return temp_path


def test_csv_export():
    """Test CSV export with and without filtering."""

    # Create test file
    test_file = create_test_file()

    # Create model
    model = HDF5TreeModel()
    model.load_file(test_file)

    # Open the file to test export
    with h5py.File(test_file, "r") as f:
        csv_group = f["test_data"]

        print("\nTest 1: Export all rows (no filtering)")
        csv_path = model._reconstruct_csv_tempfile(csv_group, "/test_data", None)
        print(f"Exported to: {csv_path}")
        with open(csv_path, 'r') as fout:
            content = fout.read()
            print("Content:")
            print(content)
            # Should have header + 5 data rows
            assert content.count('\n') == 6, f"Expected 6 lines, got {content.count(chr(10))}"

        print("\nTest 2: Export filtered rows (indices 1, 3)")
        filtered_indices = np.array([1, 3])
        csv_path = model._reconstruct_csv_tempfile(csv_group, "/test_data", filtered_indices)
        print(f"Exported to: {csv_path}")
        with open(csv_path, 'r') as fout:
            content = fout.read()
            print("Content:")
            print(content)
            # Should have header + 2 data rows
            assert content.count('\n') == 3, f"Expected 3 lines, got {content.count(chr(10))}"
            assert "Bob" in content, "Bob should be in filtered export"
            assert "David" in content, "David should be in filtered export"
            assert "Alice" not in content, "Alice should NOT be in filtered export"

        print("\nTest 3: Set and get filtered indices")
        model.set_csv_filtered_indices("/test_data", np.array([0, 2, 4]))
        indices = model.get_csv_filtered_indices("/test_data")
        print(f"Retrieved indices: {indices}")
        assert np.array_equal(indices, np.array([0, 2, 4])), "Indices don't match"

        # Clear filtered indices
        model.set_csv_filtered_indices("/test_data", None)
        indices = model.get_csv_filtered_indices("/test_data")
        print(f"Cleared indices: {indices}")
        assert indices is None, "Indices should be None after clearing"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_csv_export()
