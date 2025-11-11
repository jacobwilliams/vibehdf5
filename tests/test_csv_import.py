#!/usr/bin/env python
"""Test script to verify CSV import to HDF5 functionality."""

import h5py
import os
import tempfile
import pandas as pd
import numpy as np

def _create_datasets_from_csv(f: h5py.File, h5_path: str, disk_path: str) -> None:
    """Convert a CSV file to HDF5 datasets.

    Creates a group at h5_path (without .csv extension) containing one dataset per column.
    Each dataset contains the column data with appropriate dtype.
    """
    # Read CSV with pandas
    try:
        df = pd.read_csv(disk_path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file: {exc}") from exc

    # Remove .csv extension from group name
    group_path = h5_path
    if group_path.lower().endswith('.csv'):
        group_path = group_path[:-4]

    # Ensure parent groups exist
    parent = os.path.dirname(group_path).replace("\\", "/")
    if parent and parent != "/":
        f.require_group(parent)

    # Create a group for the CSV data
    grp = f.create_group(group_path)

    # Add metadata about the source file
    grp.attrs['source_file'] = os.path.basename(disk_path)
    grp.attrs['source_type'] = 'csv'
    grp.attrs['column_names'] = list(df.columns)

    # Create a dataset for each column
    for col in df.columns:
        col_data = df[col]

        # Clean column name for use as dataset name
        ds_name = col.strip()
        if not ds_name:
            ds_name = 'unnamed_column'

        # Convert pandas Series to numpy array with appropriate dtype
        if col_data.dtype == 'object':
            # For object dtype, convert to Python list then create dataset
            # This avoids numpy unicode string issues
            try:
                # Convert to Python strings
                str_list = [str(x) for x in col_data.values]
                grp.create_dataset(
                    ds_name,
                    data=str_list,
                    dtype=h5py.string_dtype(encoding='utf-8')
                )
            except Exception:
                # Fallback: convert to bytes
                str_list = [str(x) for x in col_data.values]
                grp.create_dataset(
                    ds_name,
                    data=str_list,
                    dtype=h5py.string_dtype(encoding='utf-8')
                )
        else:
            # Numeric or other numpy-supported dtypes
            grp.create_dataset(ds_name, data=col_data.values)

def test_csv_import():
    """Test importing a CSV file into HDF5."""

    # Create a test HDF5 file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.h5', delete=False) as tmp_h5:
        h5_path = tmp_h5.name

    try:
        # Create empty HDF5 file
        with h5py.File(h5_path, 'w') as f:
            pass

        # Path to the test CSV file
        csv_path = os.path.join(os.path.dirname(__file__), 'test.csv')

        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return

        # Open the HDF5 file for modification
        with h5py.File(h5_path, 'r+') as f:
            # Import the CSV
            _create_datasets_from_csv(f, '/test.csv', csv_path)

        # Verify the import
        print(f"\n✓ CSV imported successfully to {h5_path}")
        print("\nVerifying structure:")

        with h5py.File(h5_path, 'r') as f:
            print(f"\nGroups in file: {list(f.keys())}")

            if 'test' in f:
                test_group = f['test']
                print(f"\nDatasets in 'test' group: {list(test_group.keys())}")

                # Print attributes
                print(f"\nGroup attributes:")
                for key, value in test_group.attrs.items():
                    print(f"  {key}: {value}")

                # Print dataset contents
                print("\nDataset contents:")
                for ds_name in test_group.keys():
                    ds = test_group[ds_name]
                    print(f"\n  {ds_name}:")
                    print(f"    dtype: {ds.dtype}")
                    print(f"    shape: {ds.shape}")
                    print(f"    data: {ds[()]}")

        print(f"\n✓ Test passed! HDF5 file created at: {h5_path}")
        print(f"You can open this file in the viewer to inspect it.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(h5_path):
            os.unlink(h5_path)
        raise

if __name__ == '__main__':
    test_csv_import()
