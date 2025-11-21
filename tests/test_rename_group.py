"""Test renaming groups and datasets in the tree view."""

import os
import tempfile
import h5py
from qtpy.QtCore import Qt
from vibehdf5.hdf5_viewer import HDF5Viewer
from vibehdf5.hdf5_tree_model import HDF5TreeModel


def test_rename_group():
    """Test that renaming a group in the tree updates the HDF5 file."""
    # Create a temporary HDF5 file with a group
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create test file with a group
        with h5py.File(tmp_path, "w") as h5:
            h5.create_group("test_group")
            h5["test_group"].create_dataset("data", data=[1, 2, 3])

        # Create model and load file
        model = HDF5TreeModel()
        model.load_file(tmp_path)

        # Find the test_group item
        root = model.invisibleRootItem()
        file_item = root.child(0, 0)  # Root file item
        group_item = None
        for i in range(file_item.rowCount()):
            child = file_item.child(i, 0)
            if child and child.text() == "test_group":
                group_item = child
                break

        assert group_item is not None, "test_group not found"
        assert group_item.data(model.ROLE_PATH) == "/test_group"

        # Get the index for the group item
        group_index = model.indexFromItem(group_item)

        # Rename the group
        success = model.setData(group_index, "renamed_group", Qt.EditRole)
        assert success, "Rename should succeed"

        # Check that the item was updated
        assert group_item.text() == "renamed_group"
        assert group_item.data(model.ROLE_PATH) == "/renamed_group"

        # Verify in the HDF5 file
        with h5py.File(tmp_path, "r") as h5:
            assert "/renamed_group" in h5, "Renamed group should exist in file"
            assert "/test_group" not in h5, "Old group name should not exist"
            assert "/renamed_group/data" in h5, "Child dataset should still exist"

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_rename_dataset():
    """Test that renaming a dataset in the tree updates the HDF5 file."""
    # Create a temporary HDF5 file with a dataset
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create test file with a dataset
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset("test_dataset", data=[1, 2, 3])

        # Create model and load file
        model = HDF5TreeModel()
        model.load_file(tmp_path)

        # Find the test_dataset item
        root = model.invisibleRootItem()
        file_item = root.child(0, 0)  # Root file item
        dataset_item = None
        for i in range(file_item.rowCount()):
            child = file_item.child(i, 0)
            if child and child.text() == "test_dataset":
                dataset_item = child
                break

        assert dataset_item is not None, "test_dataset not found"
        assert dataset_item.data(model.ROLE_PATH) == "/test_dataset"

        # Get the index for the dataset item
        dataset_index = model.indexFromItem(dataset_item)

        # Rename the dataset
        success = model.setData(dataset_index, "renamed_dataset", Qt.EditRole)
        assert success, "Rename should succeed"

        # Check that the item was updated
        assert dataset_item.text() == "renamed_dataset"
        assert dataset_item.data(model.ROLE_PATH) == "/renamed_dataset"

        # Verify in the HDF5 file
        with h5py.File(tmp_path, "r") as h5:
            assert "/renamed_dataset" in h5, "Renamed dataset should exist in file"
            assert "/test_dataset" not in h5, "Old dataset name should not exist"

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_rename_nested_group():
    """Test that renaming a nested group updates paths correctly."""
    # Create a temporary HDF5 file with nested groups
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create test file with nested groups
        with h5py.File(tmp_path, "w") as h5:
            parent = h5.create_group("parent")
            child = parent.create_group("child")
            child.create_dataset("data", data=[1, 2, 3])

        # Create model and load file
        model = HDF5TreeModel()
        model.load_file(tmp_path)

        # Find the parent group item
        root = model.invisibleRootItem()
        file_item = root.child(0, 0)  # Root file item
        parent_item = None
        for i in range(file_item.rowCount()):
            child_item = file_item.child(i, 0)
            if child_item and child_item.text() == "parent":
                parent_item = child_item
                break

        assert parent_item is not None, "parent group not found"

        # Get the index for the parent item
        parent_index = model.indexFromItem(parent_item)

        # Rename the parent group
        success = model.setData(parent_index, "new_parent", Qt.EditRole)
        assert success, "Rename should succeed"

        # Check that descendant paths were updated
        child_item = None
        for i in range(parent_item.rowCount()):
            item = parent_item.child(i, 0)
            if item and item.text() == "child":
                child_item = item
                break

        assert child_item is not None, "child group not found"
        assert child_item.data(model.ROLE_PATH) == "/new_parent/child"

        # Verify in the HDF5 file
        with h5py.File(tmp_path, "r") as h5:
            assert "/new_parent" in h5, "Renamed parent group should exist"
            assert "/parent" not in h5, "Old parent name should not exist"
            assert "/new_parent/child" in h5, "Child group should exist at new path"
            assert "/new_parent/child/data" in h5, "Nested dataset should exist at new path"

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_cannot_rename_root():
    """Test that the root group cannot be renamed."""
    # Create a temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create test file
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset("data", data=[1, 2, 3])

        # Create model and load file
        model = HDF5TreeModel()
        model.load_file(tmp_path)

        # Find the root item
        root = model.invisibleRootItem()
        file_item = root.child(0, 0)  # Root file item

        assert file_item.data(model.ROLE_PATH) == "/"

        # Get the index for the root item
        root_index = model.indexFromItem(file_item)

        # Try to rename the root
        success = model.setData(root_index, "new_root", Qt.EditRole)
        assert not success, "Renaming root should fail"

        # Verify it wasn't renamed
        assert file_item.text() != "new_root"
        assert file_item.data(model.ROLE_PATH) == "/"

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    test_rename_group()
    test_rename_dataset()
    test_rename_nested_group()
    test_cannot_rename_root()
    print("All tests passed!")
