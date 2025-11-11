#!/usr/bin/env python3
"""Test GUI-based filter and export functionality."""

import h5py
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add vibehdf5 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_test_csv_hdf5():
    """Create a test HDF5 file with CSV data."""
    fd, path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)

    with h5py.File(path, 'w') as f:
        # Create a CSV-derived group
        grp = f.create_group('test_data')
        grp.attrs['source_type'] = 'csv'
        grp.attrs['source_file'] = 'test_data.csv'

        # Create datasets
        names = np.array(['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'], dtype='S10')
        ages = np.array([25, 30, 35, 40, 45, 50])
        scores = np.array([85.5, 90.0, 75.5, 88.0, 92.5, 78.0])

        grp.create_dataset('Name', data=names)
        grp.create_dataset('Age', data=ages)
        grp.create_dataset('Score', data=scores)

        # Set column metadata
        grp.attrs['column_names'] = np.array(['Name', 'Age', 'Score'], dtype='S20')
        grp.attrs['column_dataset_names'] = np.array(['Name', 'Age', 'Score'], dtype='S20')

    return path

def test_gui_workflow():
    """Test the complete GUI workflow."""
    from qtpy.QtWidgets import QApplication
    from vibehdf5.hdf5_viewer import HDF5Viewer

    # Create Qt application
    app = QApplication.instance() or QApplication(sys.argv)

    # Create test file
    test_file = create_test_csv_hdf5()
    print(f"Created test file: {test_file}")

    try:
        # Create viewer and load file
        viewer = HDF5Viewer()
        viewer.load_hdf5(test_file)

        # Find the CSV group in the tree
        model = viewer.model
        root = model.invisibleRootItem()

        # Find the test_data group
        csv_group_item = None
        for row in range(root.child(0).rowCount()):
            item = root.child(0).child(row, 0)
            if item and item.text() == 'test_data':
                csv_group_item = item
                break

        if not csv_group_item:
            print("❌ Could not find test_data group")
            return False

        # Get the index for this item
        csv_index = model.indexFromItem(csv_group_item)

        # Simulate clicking on the CSV group to display it
        viewer.tree.setCurrentIndex(csv_index)
        # Trigger selection changed manually
        sel_model = viewer.tree.selectionModel()
        if sel_model:
            viewer.on_selection_changed(sel_model.selection(), sel_model.selection())

        # Verify CSV table is displayed
        if not viewer.preview_table.isVisible():
            print("❌ CSV table not displayed")
            return False

        print(f"✅ CSV table displayed with {viewer.preview_table.rowCount()} rows")

        # Test 1: Export without filters (should export all 6 rows)
        print("\nTest 1: Export without filters")
        csv_path = viewer.tree_model.get_csv_filtered_indices('/test_data')
        print(f"Filtered indices (no filter): {csv_path}")

        # Simulate drag-and-drop export by calling _reconstruct_csv_tempfile
        with h5py.File(test_file, 'r') as f:
            grp = f['/test_data']
            exported_file = viewer.tree_model._reconstruct_csv_tempfile(grp, '/test_data')

        if exported_file:
            with open(exported_file, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                print(f"Exported {len(lines) - 1} rows (excluding header)")
                if len(lines) - 1 == 6:
                    print("✅ All rows exported correctly")
                else:
                    print(f"❌ Expected 6 rows, got {len(lines) - 1}")
                    return False

        # Test 2: Apply filters and export (Age >= 35)
        print("\nTest 2: Apply filters (Age >= 35)")
        viewer._csv_filters = [('Age', '>=', '35')]
        viewer._apply_filters()

        filtered_count = len(viewer._csv_filtered_indices)
        print(f"Filtered to {filtered_count} rows")

        if filtered_count != 4:  # Charlie, David, Eve, Frank
            print(f"❌ Expected 4 filtered rows, got {filtered_count}")
            return False

        # Verify model has the filtered indices
        model_indices = viewer.tree_model.get_csv_filtered_indices('/test_data')
        if model_indices is None:
            print("❌ Model does not have filtered indices")
            return False

        print(f"Model has filtered indices: {model_indices}")

        # Export with filters
        with h5py.File(test_file, 'r') as f:
            grp = f['/test_data']
            exported_file = viewer.tree_model._reconstruct_csv_tempfile(grp, '/test_data')

        if exported_file:
            with open(exported_file, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                print(f"Exported {len(lines) - 1} rows (excluding header)")
                print("Content:")
                print(content)

                if len(lines) - 1 == 4:
                    print("✅ Filtered rows exported correctly")
                else:
                    print(f"❌ Expected 4 rows, got {len(lines) - 1}")
                    return False

        # Test 3: Clear filters
        print("\nTest 3: Clear filters")
        viewer._clear_filters()

        model_indices = viewer.tree_model.get_csv_filtered_indices('/test_data')
        if model_indices is not None:
            print(f"❌ Model still has filtered indices after clearing: {model_indices}")
            return False

        print("✅ Filters cleared, model has no filtered indices")

        print("\n✅ All GUI workflow tests passed!")
        return True

    finally:
        # Clean up
        try:
            os.unlink(test_file)
        except:
            pass

if __name__ == '__main__':
    success = test_gui_workflow()
    sys.exit(0 if success else 1)
