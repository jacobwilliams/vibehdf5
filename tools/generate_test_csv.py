#!/usr/bin/env python3
"""Small utility to generate test CSV data with random ints, floats, and strings.

Usage examples:

# Create a file with 1000 rows and 10 columns
python tools/generate_test_csv.py --rows 1000 --cols 10 --out /tmp/test.csv

# Print CSV to stdout
python tools/generate_test_csv.py --rows 10 --cols 4

The core function is `generate_test_csv(num_rows, num_cols, ...)` which returns
either a CSV string (if `out_path` is None) or writes to `out_path` and returns
that path.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import string
import io
from typing import Optional

import numpy as np


def _random_string(length: int = 8) -> str:
    chars = string.ascii_letters + string.digits
    s = "".join(random.choices(chars, k=length))
    return s


def generate_test_csv(
    num_rows: int,
    num_cols: int,
    int_pct: float = 0.33,
    float_pct: float = 0.33,
    string_pct: float = 0.34,
    seed: Optional[int] = None,
    str_len: int = 8,
    out_path: Optional[str] = None,
    column_prefix: str = "col",
) -> str:
    """Generate a CSV containing random ints, floats, and short strings.

    Args:
        num_rows: Number of data rows (not including header).
        num_cols: Number of columns.
        int_pct: Fraction of columns that should be integers.
        float_pct: Fraction of columns that should be floats.
        string_pct: Fraction of columns that should be strings.
        seed: Optional RNG seed for reproducible output.
        str_len: Length of generated strings for string columns.
        out_path: If provided, write CSV to this path and return the path.
        column_prefix: Prefix used for column header names (e.g. col0, col1...).

    Returns:
        If `out_path` is None: string containing CSV contents.
        Otherwise: absolute path to the written CSV file.
    """
    if num_rows < 0 or num_cols < 0:
        raise ValueError("num_rows and num_cols must be >= 0")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Normalize probabilities
    total = float(int_pct + float_pct + string_pct)
    if total <= 0:
        raise ValueError("At least one of int_pct/float_pct/string_pct must be > 0")
    probs = [int_pct / total, float_pct / total, string_pct / total]

    # Choose types per column
    types = np.random.choice(["int", "float", "str"], size=num_cols, p=probs)

    # Create headers
    headers = [f"{column_prefix}{i}" for i in range(num_cols)]

    # Pre-generate numeric columns as numpy arrays for efficiency
    data_cols = [None] * num_cols

    for idx, t in enumerate(types):
        if t == "int":
            # Example integer distribution: 0..999
            data_cols[idx] = np.random.randint(0, 1000, size=num_rows)
        elif t == "float":
            # Example float distribution: normal with limited precision
            data_cols[idx] = np.round(np.random.normal(loc=0.0, scale=1.0, size=num_rows), 6)
        else:  # "str"
            data_cols[idx] = [_random_string(str_len) for _ in range(num_rows)]

    # Build rows lazily into an in-memory buffer (streaming to disk if out_path provided)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        f = open(out_path, "w", newline="")
        close_file = True
    else:
        f = io.StringIO()
        close_file = False

    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(headers)

    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            val = data_cols[c][r]
            # Ensure numpy scalar -> Python scalar to avoid weird CSV formatting
            if isinstance(val, (np.integer, np.floating)):
                val = val.item()
            row.append(val)
        writer.writerow(row)

    if close_file:
        f.close()
        return os.path.abspath(out_path)

    return f.getvalue()


def _parse_args():
    p = argparse.ArgumentParser(description="Generate test CSV data containing ints, floats, and strings.")
    p.add_argument("--rows", type=int, default=1000, help="Number of rows to generate")
    p.add_argument("--cols", type=int, default=10, help="Number of columns to generate")
    p.add_argument("--out", type=str, default=None, help="If provided, write CSV to this path")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--str-len", type=int, default=8, help="Length of generated strings")
    p.add_argument("--int-pct", type=float, default=0.33, help="Fraction of int columns")
    p.add_argument("--float-pct", type=float, default=0.33, help="Fraction of float columns")
    p.add_argument("--string-pct", type=float, default=0.34, help="Fraction of string columns")
    return p.parse_args()


def main():
    args = _parse_args()
    result = generate_test_csv(
        num_rows=args.rows,
        num_cols=args.cols,
        int_pct=args.int_pct,
        float_pct=args.float_pct,
        string_pct=args.string_pct,
        seed=args.seed,
        str_len=args.str_len,
        out_path=args.out,
    )
    if args.out:
        print(f"Wrote CSV to: {result}")
    else:
        print(result)


if __name__ == "__main__":
    main()
