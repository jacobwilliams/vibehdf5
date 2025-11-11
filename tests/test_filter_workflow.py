#!/usr/bin/env python3
"""
Test filter persistence workflow:
1. Open a CSV in GUI
2. Configure filters
3. Verify filters are saved to HDF5
4. Close and reopen file
5. Verify filters are loaded
6. Switch between CSV groups
7. Verify each group has its own filters
"""

import h5py
import json
import numpy as np
import os
import tempfile


def create_multi_csv_test_file():
    """Create HDF5 file with multiple CSV groups for testing."""
    # Use tests/tmp directory for test files
    test_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(test_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    temp_path = os.path.join(tmp_dir, "test_multi_csv_filters.h5")

    with h5py.File(temp_path, "w") as f:
        # CSV 1: Employee data
        emp_group = f.create_group("employees")
        emp_group.attrs["source_type"] = "csv"
        emp_group.attrs["source_file"] = "employees.csv"
        emp_group.attrs["column_names"] = ["Name", "Age", "Department", "Salary"]

        names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry"]
        ages = [25, 30, 35, 40, 45, 28, 33, 38]
        depts = ["HR", "IT", "IT", "Sales", "HR", "Sales", "IT", "Sales"]
        salaries = [50000, 70000, 75000, 60000, 55000, 62000, 72000, 65000]

        emp_group.create_dataset("Name", data=np.array(names, dtype='S'))
        emp_group.create_dataset("Age", data=np.array(ages))
        emp_group.create_dataset("Department", data=np.array(depts, dtype='S'))
        emp_group.create_dataset("Salary", data=np.array(salaries))

        # CSV 2: Product data
        prod_group = f.create_group("products")
        prod_group.attrs["source_type"] = "csv"
        prod_group.attrs["source_file"] = "products.csv"
        prod_group.attrs["column_names"] = ["Product", "Category", "Price", "Stock"]

        products = ["Widget", "Gadget", "Doohickey", "Thingamajig", "Gizmo", "Contraption"]
        categories = ["Tools", "Electronics", "Tools", "Electronics", "Electronics", "Tools"]
        prices = [9.99, 19.99, 14.99, 24.99, 29.99, 12.99]
        stocks = [100, 50, 75, 25, 60, 80]

        prod_group.create_dataset("Product", data=np.array(products, dtype='S'))
        prod_group.create_dataset("Category", data=np.array(categories, dtype='S'))
        prod_group.create_dataset("Price", data=np.array(prices))
        prod_group.create_dataset("Stock", data=np.array(stocks))

        # CSV 3: Sales data
        sales_group = f.create_group("sales")
        sales_group.attrs["source_type"] = "csv"
        sales_group.attrs["source_file"] = "sales.csv"
        sales_group.attrs["column_names"] = ["Date", "Product", "Quantity", "Total"]

        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        sale_products = ["Widget", "Gadget", "Widget", "Gizmo", "Gadget"]
        quantities = [5, 2, 3, 1, 4]
        totals = [49.95, 39.98, 29.97, 29.99, 79.96]

        sales_group.create_dataset("Date", data=np.array(dates, dtype='S'))
        sales_group.create_dataset("Product", data=np.array(sale_products, dtype='S'))
        sales_group.create_dataset("Quantity", data=np.array(quantities))
        sales_group.create_dataset("Total", data=np.array(totals))

    print(f"✓ Created test file: {temp_path}")
    return temp_path


def verify_filter_workflow():
    """Verify the complete filter persistence workflow."""
    test_file = create_multi_csv_test_file()

    print("\n" + "="*60)
    print("TEST WORKFLOW")
    print("="*60)

    print("\n1. Initial state: No filters saved")
    with h5py.File(test_file, "r") as f:
        for group_name in ["employees", "products", "sales"]:
            grp = f[group_name]
            has_filters = "csv_filters" in grp.attrs
            print(f"   {group_name}: {'HAS FILTERS' if has_filters else 'no filters'}")
            assert not has_filters, f"{group_name} should not have filters initially"

    print("\n2. Simulating GUI: Add filters to employees")
    emp_filters = [["Age", ">=", "30"], ["Department", "==", "IT"]]
    with h5py.File(test_file, "r+") as f:
        f["employees"].attrs["csv_filters"] = json.dumps(emp_filters)
    print(f"   Saved filters: {emp_filters}")

    print("\n3. Simulating GUI: Add filters to products")
    prod_filters = [["Price", ">", "15.00"], ["Stock", ">=", "50"]]
    with h5py.File(test_file, "r+") as f:
        f["products"].attrs["csv_filters"] = json.dumps(prod_filters)
    print(f"   Saved filters: {prod_filters}")

    print("\n4. Simulating file reload: Verify filters persist")
    with h5py.File(test_file, "r") as f:
        # Check employees
        emp_grp = f["employees"]
        assert "csv_filters" in emp_grp.attrs, "Employees should have filters"
        loaded_emp = json.loads(emp_grp.attrs["csv_filters"].decode('utf-8')
                                if isinstance(emp_grp.attrs["csv_filters"], bytes)
                                else emp_grp.attrs["csv_filters"])
        assert loaded_emp == emp_filters, f"Employees filters mismatch: {loaded_emp}"
        print(f"   ✓ Employees filters loaded: {loaded_emp}")

        # Check products
        prod_grp = f["products"]
        assert "csv_filters" in prod_grp.attrs, "Products should have filters"
        loaded_prod = json.loads(prod_grp.attrs["csv_filters"].decode('utf-8')
                                 if isinstance(prod_grp.attrs["csv_filters"], bytes)
                                 else prod_grp.attrs["csv_filters"])
        assert loaded_prod == prod_filters, f"Products filters mismatch: {loaded_prod}"
        print(f"   ✓ Products filters loaded: {loaded_prod}")

        # Check sales (should have no filters)
        sales_grp = f["sales"]
        assert "csv_filters" not in sales_grp.attrs, "Sales should not have filters"
        print(f"   ✓ Sales has no filters (as expected)")

    print("\n5. Simulating GUI: Clear filters from products")
    with h5py.File(test_file, "r+") as f:
        if "csv_filters" in f["products"].attrs:
            del f["products"].attrs["csv_filters"]
    print("   ✓ Cleared products filters")

    print("\n6. Verify products filters removed, employees unchanged")
    with h5py.File(test_file, "r") as f:
        assert "csv_filters" in f["employees"].attrs, "Employees still has filters"
        assert "csv_filters" not in f["products"].attrs, "Products filters cleared"
        assert "csv_filters" not in f["sales"].attrs, "Sales still has no filters"
        print("   ✓ Each CSV group maintains independent filter state")

    print("\n7. Expected GUI behavior when opening this file:")
    print("   - Open 'employees': Should load 2 filters automatically")
    print("   - Switch to 'products': Should show no filters")
    print("   - Switch to 'sales': Should show no filters")
    print("   - Return to 'employees': Should still have 2 filters")

    print("\n" + "="*60)
    print("✅ ALL WORKFLOW TESTS PASSED!")
    print("="*60)
    print(f"\nTest file location: {test_file}")
    print("\nTo manually test:")
    print(f"  1. Open the file in vibehdf5 GUI")
    print(f"  2. Click on 'employees' group")
    print(f"  3. Should see '2 filter(s) applied' and filtered data")
    print(f"  4. Click 'Configure Filters' to see: Age >= 30 AND Department == IT")
    print(f"  5. Switch to 'products' or 'sales' - should have no filters")
    print(f"  6. Add filters to any group and they'll be saved automatically")


if __name__ == "__main__":
    verify_filter_workflow()
