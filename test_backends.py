"""
Test script to verify backend implementations.

This script tests both HDF5 and SQLite backends to ensure they work correctly.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from vibehdf5.backend_factory import create_backend


def test_backend(backend_type: str):
    """Test a storage backend.

    Args:
        backend_type: "hdf5" or "sqlite"
    """
    print(f"\n{'=' * 60}")
    print(f"Testing {backend_type.upper()} Backend")
    print('=' * 60)

    # Create temporary file
    if backend_type == "hdf5":
        ext = ".h5"
    else:
        ext = ".db"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        filepath = f.name

    try:
        # Create backend
        backend = create_backend(filepath, backend_type)
        backend.create(filepath)

        print(f"✓ Created {backend_type} file: {filepath}")

        # Test 1: Create groups
        backend.create_group("/data")
        backend.create_group("/data/subgroup")
        print("✓ Created groups")

        # Test 2: Write datasets
        backend.write_dataset("/data/array", np.array([1, 2, 3, 4, 5]))
        backend.write_dataset("/data/matrix", np.array([[1, 2], [3, 4]]))
        backend.write_dataset("/data/text", "Hello, World!")
        backend.write_dataset("/data/subgroup/values", np.array([10, 20, 30]), compression="gzip")
        print("✓ Created datasets")

        # Test 3: Set attributes
        backend.set_attribute("/data", "description", "Test data group")
        backend.set_attribute("/data/array", "units", "meters")
        backend.set_attribute("/data/array", "scale", 1.5)
        backend.set_attribute("/data/array", "flag", True)
        print("✓ Set attributes")

        # Test 4: Read data
        array_data = backend.read_dataset("/data/array")
        assert np.array_equal(array_data, np.array([1, 2, 3, 4, 5])), "Array data mismatch"

        matrix_data = backend.read_dataset("/data/matrix")
        assert np.array_equal(matrix_data, np.array([[1, 2], [3, 4]])), "Matrix data mismatch"

        text_data = backend.read_dataset("/data/text")
        assert "Hello" in str(text_data), "Text data mismatch"

        print("✓ Read datasets correctly")

        # Test 5: Read attributes
        attrs = backend.get_attributes("/data")
        assert attrs["description"] == "Test data group", "Attribute mismatch"

        array_attrs = backend.get_attributes("/data/array")
        assert array_attrs["units"] == "meters", "Units attribute mismatch"
        assert abs(array_attrs["scale"] - 1.5) < 0.01, "Scale attribute mismatch"
        assert array_attrs["flag"] == True, "Flag attribute mismatch"  # noqa: E712

        print("✓ Read attributes correctly")

        # Test 6: List children
        children = backend.list_children("/data")
        child_names = [c.name for c in children]
        assert "array" in child_names, "Missing array in children"
        assert "subgroup" in child_names, "Missing subgroup in children"
        print(f"✓ Listed children: {child_names}")

        # Test 7: Node existence
        assert backend.exists("/data/array"), "Node should exist"
        assert not backend.exists("/data/nonexistent"), "Node should not exist"
        print("✓ Node existence checks work")

        # Test 8: Create CSV group
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [95.5, 87.2, 91.8]
        })
        backend.create_csv_group("/csv_data", df)
        print("✓ Created CSV group")

        # Test 9: Read CSV data
        df_read = backend.get_csv_dataframe("/csv_data")
        assert len(df_read) == 3, "CSV row count mismatch"
        assert list(df_read.columns) == ["name", "age", "score"], "CSV columns mismatch"
        print("✓ Read CSV data correctly")

        # Test 10: CSV filters and sorts
        backend.set_csv_filtered_indices("/csv_data", [0, 2])
        filtered = backend.get_csv_filtered_indices("/csv_data")
        assert filtered == [0, 2], "CSV filter mismatch"

        backend.set_csv_sort_spec("/csv_data", [("age", False)])
        sort_spec = backend.get_csv_sort_spec("/csv_data")
        assert sort_spec == [("age", False)], "CSV sort spec mismatch"

        print("✓ CSV filters and sorts work")

        # Test 11: Rename node
        backend.rename_node("/data/array", "/data/renamed_array")
        assert backend.exists("/data/renamed_array"), "Renamed node should exist"
        assert not backend.exists("/data/array"), "Old node should not exist"
        print("✓ Renamed node")

        # Test 12: Copy node
        backend.copy_node("/data/renamed_array", "/data/copied_array")
        assert backend.exists("/data/copied_array"), "Copied node should exist"
        copied_data = backend.read_dataset("/data/copied_array")
        assert np.array_equal(copied_data, np.array([1, 2, 3, 4, 5])), "Copied data mismatch"
        print("✓ Copied node")

        # Test 13: Delete node
        backend.delete_node("/data/copied_array")
        assert not backend.exists("/data/copied_array"), "Deleted node should not exist"
        print("✓ Deleted node")

        # Test 14: Export CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name
        try:
            backend.export_csv_group("/csv_data", csv_path)
            df_exported = pd.read_csv(csv_path)
            assert len(df_exported) == 3, "Exported CSV row count mismatch"
            print("✓ Exported CSV group")
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

        # Close backend
        backend.close()
        print("✓ Closed backend")

        print(f"\n✅ All tests passed for {backend_type.upper()} backend!")

    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✓ Cleaned up test file: {filepath}")


def main():
    """Run tests for all backends."""
    print("\n" + "=" * 60)
    print("Backend Implementation Tests")
    print("=" * 60)

    # Test HDF5 backend
    test_backend("hdf5")

    # Test SQLite backend
    test_backend("sqlite")

    print("\n" + "=" * 60)
    print("✅ ALL BACKEND TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
