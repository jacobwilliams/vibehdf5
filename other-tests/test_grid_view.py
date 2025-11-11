#!/usr/bin/env python
"""Quick test to create an HDF5 file with CSV data and open it in the viewer."""

import h5py
import sys
from pathlib import Path
import os
import sys

# Add vibehdf5 to path
# sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vibehdf5.hdf5_viewer import main

# Create a test HDF5 file with CSV-like structure
test_file = "test_grid_view.h5"

with h5py.File(test_file, "w") as f:
    # Create a CSV-derived group
    grp = f.create_group("test_csv_data")
    grp.attrs['source_file'] = 'test.csv'
    grp.attrs['source_type'] = 'csv'
    grp.attrs['column_names'] = ['time string', 'et', 'x']

    # Add datasets
    grp.create_dataset('time string',
                       data=["jan 1, 2000", "Jan 2, 2000", "Jan 3, 2000"],
                       dtype=h5py.string_dtype(encoding='utf-8'))
    grp.create_dataset('et', data=[0.0, 3600.0, 8900.0])
    grp.create_dataset('x', data=[1.0, 2.0, 3.0])

print(f"Created test file: {test_file}")
print("Opening in viewer...")

# Launch the viewer with this file
sys.argv = ['test', test_file]
sys.exit(main())
