#!/usr/bin/env python
"""
Test script to demonstrate syntax highlighting in the HDF5 viewer.
Creates an HDF5 file with various text datasets in different languages.
"""

import tempfile
import os

def create_test_file():
    """Create an HDF5 file with sample code in various languages."""
    print("Creating test HDF5 file with syntax highlighting examples...")
    print("=" * 70)

    try:
        import h5py
    except ImportError:
        print("Error: h5py not installed. Install with: pip install h5py")
        return None

    # Sample code snippets
    samples = {
        "example.py": '''#!/usr/bin/env python
"""Sample Python script for testing syntax highlighting."""

import sys
from pathlib import Path

class DataProcessor:
    """Process data with various methods."""

    def __init__(self, name: str):
        self.name = name
        self.count = 0

    def process(self, data: list[int]) -> int:
        """Process a list of integers."""
        # Calculate sum and average
        total = sum(data)
        avg = total / len(data) if data else 0

        print(f"Processing {len(data)} items")
        print(f"Total: {total}, Average: {avg}")

        return total

if __name__ == "__main__":
    processor = DataProcessor("test")
    result = processor.process([1, 2, 3, 4, 5])
    print(f"Result: {result}")
''',
        "script.js": '''// Sample JavaScript code
const API_URL = "https://api.example.com";

class UserManager {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.users = [];
    }

    async fetchUsers() {
        try {
            const response = await fetch(this.apiUrl + '/users');
            this.users = await response.json();
            return this.users;
        } catch (error) {
            console.error("Failed to fetch users:", error);
            return [];
        }
    }

    getUserById(id) {
        return this.users.find(user => user.id === id);
    }
}

// Usage
const manager = new UserManager(API_URL);
manager.fetchUsers().then(users => {
    console.log(`Loaded ${users.length} users`);
});
''',
        "config.json": '''{
    "name": "vibehdf5",
    "version": "1.0.2",
    "description": "HDF5 file viewer and manager",
    "main": "hdf5_viewer.py",
    "dependencies": {
        "h5py": "^3.0.0",
        "numpy": "^1.20.0",
        "PySide6": "^6.0.0"
    },
    "features": {
        "syntax_highlighting": true,
        "drag_drop": true,
        "image_preview": true
    },
    "supported_languages": [
        "python",
        "javascript",
        "json",
        "yaml",
        "xml",
        "html",
        "css"
    ]
}
''',
        "settings.yaml": '''# Application settings
app:
  name: vibehdf5
  version: 1.0.2
  debug: false

database:
  path: ./data/files.h5
  auto_backup: true
  backup_interval: 3600

ui:
  theme: light
  font_size: 12
  show_line_numbers: true
  syntax_highlighting:
    enabled: true
    languages:
      - python
      - javascript
      - json
      - yaml

features:
  drag_drop: true
  auto_save: false
  preview:
    max_size: 1000000  # bytes
    image_formats:
      - png
      - jpg
''',
        "example.cpp": '''// Sample C++ code with classes and templates
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

template<typename T>
class DataContainer {
private:
    std::vector<T> data;
    std::string name;

public:
    DataContainer(const std::string& n) : name(n) {}

    void add(const T& item) {
        data.push_back(item);
    }

    size_t size() const {
        return data.size();
    }

    T max() const {
        return *std::max_element(data.begin(), data.end());
    }
};

int main() {
    DataContainer<int> numbers("test");
    numbers.add(10);
    numbers.add(25);
    numbers.add(15);

    std::cout << "Size: " << numbers.size() << std::endl;
    std::cout << "Max: " << numbers.max() << std::endl;

    return 0;
}
''',
        "program.f90": '''! Sample Fortran 90 code
program matrix_operations
    implicit none

    integer, parameter :: n = 100
    real, dimension(n,n) :: A, B, C
    real :: alpha = 2.0
    integer :: i, j

    ! Initialize matrices
    do i = 1, n
        do j = 1, n
            A(i,j) = real(i + j)
            B(i,j) = real(i * j)
        end do
    end do

    ! Matrix addition
    C = A + alpha * B

    ! Print diagonal elements
    write(*,*) "Diagonal elements:"
    do i = 1, min(10, n)
        write(*,'(A,I0,A,F10.2)') "C(", i, ",", i, ") = ", C(i,i)
    end do

end program matrix_operations
''',
        "config.toml": '''# TOML configuration file
[project]
name = "vibehdf5"
version = "1.0.2"
authors = ["Jacob Williams"]
license = "MIT"

[dependencies]
h5py = "^3.0.0"
numpy = "^1.20.0"
PySide6 = "^6.0.0"

[features]
syntax-highlighting = true
drag-drop = true
image-preview = true

[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"
''',
        "README.md": '''# Syntax Highlighting Test

## Overview

This file tests **Markdown** syntax highlighting with various elements:

### Code Blocks

```python
def hello_world():
    print("Hello, World!")
```

### Lists

- Item 1
- Item 2
  - Nested item
  - Another nested item

### Links and Emphasis

Check out [vibehdf5](https://github.com/jacobwilliams/vibehdf5) for more info!

Use *italics* and **bold** for emphasis.

### Inline Code

Use `import h5py` to work with HDF5 files in Python.
''',
    }

    # Create temp file in tests/tmp directory
    test_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(test_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    temp_file = os.path.join(tmp_dir, "syntax_highlight_test.h5")

    with h5py.File(temp_file, "w") as f:
        # Create a group for examples
        examples_group = f.create_group("code_examples")

        # Add all sample files
        for filename, content in samples.items():
            dataset_name = f"code_examples/{filename}"
            f.create_dataset(dataset_name, data=content, dtype=h5py.string_dtype(encoding='utf-8'))
            print(f"✓ Added {filename} ({len(content)} bytes)")

        # Add metadata
        f.attrs["description"] = "Test file for syntax highlighting"
        f.attrs["created_by"] = "test_syntax_highlighting.py"
        f.attrs["num_examples"] = len(samples)

    print("\n" + "=" * 70)
    print(f"✓ Test file created: {temp_file}")
    print(f"  Contains {len(samples)} code samples in different languages")
    print("\nTo test:")
    print(f"  1. Open the file in vibehdf5:")
    print(f"     vibehdf5 {temp_file}")
    print(f"  2. Navigate to code_examples group")
    print(f"  3. Click on each file to see syntax highlighting")
    print("\nSupported languages:")
    langs = set()
    for filename in samples.keys():
        ext = filename.split('.')[-1] if '.' in filename else ''
        if ext:
            langs.add(ext)
    print(f"  {', '.join(sorted(langs))}")

    return temp_file

def test_create_file():
    """Test creating the syntax highlighting file."""
    test_file = create_test_file()
    if test_file:
        print(f"\n✓ Test file ready at: {test_file}")
        print("Open it with: vibehdf5 " + test_file)
        assert True, "Test file created successfully"
    else:
        assert False, "Failed to create test file"