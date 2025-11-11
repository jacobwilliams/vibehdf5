#!/usr/bin/env python3
"""Test script to verify CSV filter persistence in HDF5 files."""

import h5py
import json
import numpy as np
import os
import tempfile


def create_test_file_with_filters():
    """Create a test HDF5 file with CSV data and saved filters."""
    # Use tests/tmp directory for test files
    test_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(test_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    temp_path = os.path.join(tmp_dir, "test_filter_persistence.h5")

    with h5py.File(temp_path, "w") as f:
        # Create first CSV group
        csv_group1 = f.create_group("employees")
        csv_group1.attrs["source_type"] = "csv"
        csv_group1.attrs["source_file"] = "employees.csv"
        csv_group1.attrs["column_names"] = ["Name", "Age", "Department", "Salary"]

        # Add data
        names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"]
        ages = [25, 30, 35, 40, 45, 28, 33]
        departments = ["HR", "IT", "IT", "Sales", "HR", "Sales", "IT"]
        salaries = [50000, 70000, 75000, 60000, 55000, 62000, 72000]

        csv_group1.create_dataset("Name", data=np.array(names, dtype='S'))
        csv_group1.create_dataset("Age", data=np.array(ages))
        csv_group1.create_dataset("Department", data=np.array(departments, dtype='S'))
        csv_group1.create_dataset("Salary", data=np.array(salaries))

        # Save filters: Age >= 30 AND Department == IT
        filters = [
            ["Age", ">=", "30"],
            ["Department", "==", "IT"]
        ]
        csv_group1.attrs["csv_filters"] = json.dumps(filters)

        # Create second CSV group (no filters)
        csv_group2 = f.create_group("products")
        csv_group2.attrs["source_type"] = "csv"
        csv_group2.attrs["source_file"] = "products.csv"
        csv_group2.attrs["column_names"] = ["Product", "Price", "Stock"]

        products = ["Widget", "Gadget", "Doohickey", "Thingamajig"]
        prices = [9.99, 19.99, 14.99, 24.99]
        stocks = [100, 50, 75, 25]

        csv_group2.create_dataset("Product", data=np.array(products, dtype='S'))
        csv_group2.create_dataset("Price", data=np.array(prices))
        csv_group2.create_dataset("Stock", data=np.array(stocks))

    print(f"Created test file: {temp_path}")
    return temp_path


def test_filter_persistence():
    """Test that filters are persisted and loaded correctly."""
    test_file = create_test_file_with_filters()

    print("\n=== Test 1: Verify filters saved in HDF5 ===")
    with h5py.File(test_file, "r") as f:
        employees_group = f["employees"]
        products_group = f["products"]

        # Check employees group has filters
        assert "csv_filters" in employees_group.attrs, "Employees should have csv_filters attribute"
        filters_json = employees_group.attrs["csv_filters"]
        if isinstance(filters_json, bytes):
            filters_json = filters_json.decode('utf-8')
        filters = json.loads(filters_json)
        print(f"Employees filters: {filters}")
        assert len(filters) == 2, f"Expected 2 filters, got {len(filters)}"
        assert filters[0] == ["Age", ">=", "30"], f"First filter mismatch: {filters[0]}"
        assert filters[1] == ["Department", "==", "IT"], f"Second filter mismatch: {filters[1]}"

        # Check products group has no filters
        assert "csv_filters" not in products_group.attrs, "Products should NOT have csv_filters attribute"
        print("Products filters: None (as expected)")

    print("\n=== Test 2: Test saving new filters ===")
    with h5py.File(test_file, "r+") as f:
        products_group = f["products"]

        # Add filters to products
        new_filters = [["Price", ">", "15.00"]]
        products_group.attrs["csv_filters"] = json.dumps(new_filters)
        print(f"Saved new filters to products: {new_filters}")

    # Verify saved
    with h5py.File(test_file, "r") as f:
        products_group = f["products"]
        assert "csv_filters" in products_group.attrs, "Products should now have csv_filters"
        filters_json = products_group.attrs["csv_filters"]
        if isinstance(filters_json, bytes):
            filters_json = filters_json.decode('utf-8')
        filters = json.loads(filters_json)
        print(f"Loaded filters from products: {filters}")
        assert filters == new_filters, f"Filter mismatch: {filters} != {new_filters}"

    print("\n=== Test 3: Test clearing filters ===")
    with h5py.File(test_file, "r+") as f:
        employees_group = f["employees"]

        # Clear filters by deleting attribute
        if "csv_filters" in employees_group.attrs:
            del employees_group.attrs["csv_filters"]
            print("Cleared filters from employees")

    # Verify cleared
    with h5py.File(test_file, "r") as f:
        employees_group = f["employees"]
        assert "csv_filters" not in employees_group.attrs, "Employees should no longer have csv_filters"
        print("Confirmed: employees has no filters")

    print("\n✅ All filter persistence tests passed!")
    print(f"\nTest file location: {test_file}")
    print("You can open this file in the GUI to verify filter loading works correctly.")
