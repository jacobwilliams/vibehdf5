#!/usr/bin/env python
"""
Test script to demonstrate the new file creation feature.
This simulates what happens when the user clicks "New HDF5 File…" in the GUI.
"""

import tempfile
import os

def test_new_file_creation():
    """Test creating a new HDF5 file and adding content to it."""
    print("Testing New HDF5 File Feature")
    print("=" * 50)
    
    # Create a temporary file path in tests/tmp directory
    test_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(test_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    test_file = os.path.join(tmp_dir, "test_vibehdf5_new.h5")
    
    print(f"\n1. Creating new HDF5 file: {test_file}")
    
    try:
        import h5py
        
        # Step 1: Create new empty file (what the GUI does)
        with h5py.File(test_file, "w"):
            pass
        print("   ✓ Empty HDF5 file created successfully")
        
        # Step 2: Verify we can open and read it
        with h5py.File(test_file, "r") as f:
            print(f"   ✓ File opened successfully")
            print(f"   - Root groups: {list(f.keys())}")
            print(f"   - Root attributes: {list(f.attrs.keys())}")
        
        # Step 3: Add some content (simulating drag-drop or add files)
        print("\n2. Adding content to the file...")
        with h5py.File(test_file, "a") as f:
            # Create a group
            grp = f.create_group("my_data")
            grp.attrs["description"] = "Test data group"
            
            # Add datasets
            grp.create_dataset("numbers", data=[1, 2, 3, 4, 5])
            grp.create_dataset("text", data="Hello from vibehdf5!")
            
            # Add file-level attribute
            f.attrs["created_by"] = "vibehdf5"
            f.attrs["version"] = "1.0"
        
        print("   ✓ Content added successfully")
        
        # Step 4: Verify the content
        print("\n3. Verifying file structure...")
        with h5py.File(test_file, "r") as f:
            print(f"   Root groups: {list(f.keys())}")
            print(f"   Root attributes: {dict(f.attrs)}")
            
            if "my_data" in f:
                grp = f["my_data"]
                print(f"   'my_data' group contains: {list(grp.keys())}")
                print(f"   'my_data' attributes: {dict(grp.attrs)}")
                
                if "numbers" in grp:
                    print(f"   'numbers' dataset: {grp['numbers'][()]}")
                if "text" in grp:
                    print(f"   'text' dataset: {grp['text'][()]}")
        
        print("\n✓ All tests passed!")
        print("\nIn the GUI:")
        print("  1. Click 'New HDF5 File…' button or press Ctrl+N")
        print("  2. Choose a location and filename")
        print("  3. The new file is created and loaded in the viewer")
        print("  4. You can immediately start adding files and folders via:")
        print("     - 'Add Files…' button (Ctrl+Shift+F)")
        print("     - 'Add Folder…' button (Ctrl+Shift+D)")
        print("     - Drag & drop from your file manager")
        
    except ImportError:
        print("   ✗ h5py not installed (install via: pip install h5py)")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n4. Test file cleaned up: {test_file}")
    
    return True

if __name__ == "__main__":
    success = test_new_file_creation()
    exit(0 if success else 1)
