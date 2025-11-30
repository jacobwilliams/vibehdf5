"""
Utilities for working with HDF5 files in VibeHDF5.
"""

import os
import h5py
import fnmatch
import numpy as np
from typing import Union
import gzip
import binascii

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
    """Archive all files in a directory (and subdirectories) matching a file pattern into an hdf5 file.

    Args:
        directory: Path to the directory to archive.
        hdf5_filename: Path to the output HDF5 file.
        file_pattern: A glob pattern or list of patterns to match files (default: "*.*").
        verbose: If True, print the names of files being archived.

    Warning:
        This is an old function, and may not properly add the data in the same
        way as the main GUI. Use with caution.
    """

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
    """print the file structure stored in an hdf5 file
    Args:
        hdf5_filename: Path to the HDF5 file.
    """

    fin = h5py.File(hdf5_filename, "r")

    def print_attrs(name, obj):
        print(f"{name}: {obj}")

    fin.visititems(print_attrs)

    fin.close()


def indices_to_ranges(indices: list[int] | np.ndarray) -> list[str | int]:
    """Convert a list of indices to a compact range representation.

    Consecutive indices are represented as 'start-end' strings, while
    isolated indices remain as integers.

    Args:
        indices: List or array of sorted integers

    Returns:
        List of range strings and/or integers

    Examples:
        [1,2,3,4,5,10] -> ['1-5', 10]
        [1,3,5,7,9] -> [1, 3, 5, 7, 9]
        [1,2,3,10,11,12,20] -> ['1-3', '10-12', 20]
    """
    if not len(indices):
        return []

    if isinstance(indices, np.ndarray):
        indices = indices.tolist()

    result = []
    start = indices[0]
    end = indices[0]

    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            # Extend the current range
            end = indices[i]
        else:
            # Save the current range and start a new one
            if end > start:
                result.append(f"{start}-{end}")
            else:
                result.append(start)
            start = indices[i]
            end = indices[i]

    # Save the last range
    if end > start:
        result.append(f"{start}-{end}")
    else:
        result.append(start)

    return result


def ranges_to_indices(ranges: list[str | int]) -> np.ndarray:
    """Convert a compact range representation back to a list of indices.

    Args:
        ranges: List of range strings and/or integers

    Returns:
        Numpy array of integers

    Examples:
        ['1-5', 10] -> [1,2,3,4,5,10]
        [1, 3, 5, 7, 9] -> [1,3,5,7,9]
        ['1-3', '10-12', 20] -> [1,2,3,10,11,12,20]
    """
    if not ranges:
        return np.array([], dtype=np.int64)

    indices = []
    for item in ranges:
        if isinstance(item, str) and "-" in item:
            # Parse range string
            start_str, end_str = item.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            indices.extend(range(start, end + 1))
        else:
            # Single index
            indices.append(int(item))

    return np.array(indices, dtype=np.int64)


def sanitize_hdf5_name(name: str) -> str:
    """Sanitize a name for use as an HDF5 dataset/group member.

    - replaces '/' with '_' (since '/' is the HDF5 path separator)
    - strips leading/trailing whitespace
    - returns 'unnamed' if the result is empty
    """
    try:
        s = (name or "").strip()
        s = s.replace("/", "_")
        return s or "unnamed"
    except Exception:  # noqa: BLE001
        return "unnamed"


def dataset_to_text(ds, limit_bytes: int = 1_000_000) -> tuple[str, str | None]:
    """Read an h5py dataset and return a text representation and an optional note.

    - If content exceeds limit_bytes, the output is truncated with a note.
    - Tries to decode bytes as UTF-8; falls back to hex preview for binary.
    - Automatically decompresses gzip-compressed text datasets.
    """

    note = None
    # Best effort: read entire dataset (beware huge data)
    data = ds[()]

    # Check if this is a gzip-compressed text dataset
    try:
        if "compressed" in ds.attrs and ds.attrs["compressed"] == "gzip":
            if isinstance(data, np.ndarray) and data.dtype == np.uint8:
                compressed_bytes = data.tobytes()
                decompressed = gzip.decompress(compressed_bytes)
                encoding = ds.attrs.get("original_encoding", "utf-8")
                if isinstance(encoding, bytes):
                    encoding = encoding.decode("utf-8")
                # Check if this is binary data
                if encoding == "binary":
                    # Return decompressed binary data for further processing
                    return _bytes_to_text(decompressed, limit_bytes, decompressed=True)
                # Otherwise it's text
                text = decompressed.decode(encoding)
                if len(text) > limit_bytes:
                    text = text[:limit_bytes] + "\n… (truncated)"
                    note = f"Preview limited to {limit_bytes} characters (decompressed)"
                else:
                    note = "(decompressed from gzip)"
                return text, note
    except Exception:  # noqa: BLE001
        pass

    # Convert to bytes if it's an array of fixed-length ASCII (S) blocks
    if isinstance(data, np.ndarray) and data.dtype.kind == "S":
        try:
            # Flatten and join bytes chunks
            b = b"".join(x.tobytes() if hasattr(x, "tobytes") else bytes(x) for x in data.ravel())
        except Exception:
            b = data.tobytes()
        return _bytes_to_text(b, limit_bytes)

    # Variable length strings
    vld = h5py.check_string_dtype(ds.dtype)
    if vld is not None:
        try:
            # Read as Python str
            as_str = ds.asstr()[()]
            if isinstance(as_str, np.ndarray):
                text = "\n".join(map(str, as_str.ravel().tolist()))
            else:
                text = str(as_str)
            note = None
            if len(text.encode("utf-8")) > limit_bytes:
                enc = text.encode("utf-8")[:limit_bytes]
                text = enc.decode("utf-8", errors="ignore") + "\n… (truncated)"
                note = f"Preview limited to {limit_bytes} bytes"
            return text, note
        except Exception:
            pass

    # Raw bytes
    if isinstance(data, (bytes, bytearray, np.void)):
        return _bytes_to_text(bytes(data), limit_bytes)

    # Numeric or other arrays: show a compact preview
    if isinstance(data, np.ndarray):
        flat = data.ravel()
        preview_count = min(2000, flat.size)
        text = np.array2string(flat[:preview_count], threshold=preview_count)
        note = None
        if flat.size > preview_count:
            note = f"Showing first {preview_count} elements out of {flat.size}"
        return text, note

    # Fallback to repr
    t = repr(data)
    if len(t) > 200_000:
        t = t[:200_000] + "… (truncated)"
        note = "Preview truncated"
    return t, note


def _bytes_to_text(
    b: bytes, limit_bytes: int = 1_000_000, decompressed: bool = False
) -> tuple[str, str | None]:
    """
    Convert bytes to a text representation, decoding as UTF-8 if possible, or as hex if not.

    Args:
        b: Bytes object to convert.
        limit_bytes: Maximum number of bytes to preview (truncates if exceeded).
        decompressed: True if bytes were decompressed from gzip (adds note).

    Returns:
        Tuple of (text, note) where text is the decoded string or hex preview, and note is an optional string describing truncation or decompression.
    """
    note = None
    if len(b) > limit_bytes:
        b = b[:limit_bytes]
        note = f"Preview limited to {limit_bytes} bytes"
        if decompressed:
            note = f"Preview limited to {limit_bytes} bytes (decompressed)"
    elif decompressed:
        note = "(decompressed from gzip)"
    try:
        return b.decode("utf-8"), note
    except UnicodeDecodeError:
        # Provide a hex dump preview
        hexstr = binascii.hexlify(b).decode("ascii")
        # Group hex bytes in pairs for readability
        grouped = " ".join(hexstr[i : i + 2] for i in range(0, len(hexstr), 2))
        if len(grouped) > 200_000:
            grouped = grouped[:200_000] + "… (truncated)"
            note = "Preview truncated"
        suffix = " (decompressed)" if decompressed else ""
        return grouped, ((note or "binary data shown as hex") + suffix)
