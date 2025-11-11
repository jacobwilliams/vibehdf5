import os
import h5py
import fnmatch
import numpy as np
from typing import Union

excluded_dirs = [".git", ".svn"]  # never include these subdirectories
excluded_files = [
    ".DS_Store",
    ".DS_Store?",
    ".Spotlight-V100",
    ".Trashes",
    "Thumbs.db",
]  # never include these files


def archive_to_hdf5(
    directory: str,
    hdf5_filename: str,
    file_pattern: Union[str, list[str]] = "*.*",
    verbose: bool = False,
):
    """Archive all files in a directory (and subdirectories) matching a file pattern into an hdf5 file."""

    if not isinstance(file_pattern, list):
        file_pattern = [file_pattern]

    fout = h5py.File(hdf5_filename, "w")

    # walk the directory structure:
    for dirpath, dirnames, filenames in os.walk(directory):
        # get all files that match the pattern:
        for pattern in file_pattern:
            for filename in fnmatch.filter(filenames, pattern):
                if filename in excluded_files:
                    continue
                if any(i in excluded_dirs for i in dirnames):
                    continue
                if verbose:
                    print(os.path.join(dirpath, filename))
                relpath = os.path.relpath(os.path.join(dirpath, filename), directory)
                dir_name = os.path.split(directory)[-1]
                path_for_file = os.path.join(dir_name, relpath).replace("\\", "/")
                name = os.path.join(dirpath, filename).replace("\\", "/")

                try:
                    # try to save file contents as a string:
                    with open(name, "r", encoding="utf-8") as f:
                        data = f.read()
                    fout.create_dataset(
                        path_for_file, data=data, dtype=h5py.string_dtype(encoding="utf-8")
                    )
                except Exception:
                    # Save as binary: store as 1D uint8 array for compatibility
                    with open(name, "rb") as f:
                        bdata = f.read()
                    fout.create_dataset(path_for_file, data=np.frombuffer(bdata, dtype="uint8"))

    fout.close()


def print_file_structure_in_hdf5(hdf5_filename: str):
    """print the file structure stored in an hdf5 file"""

    fin = h5py.File(hdf5_filename, "r")

    def print_attrs(name, obj):
        print(f"{name}: {obj}")

    fin.visititems(print_attrs)

    fin.close()
