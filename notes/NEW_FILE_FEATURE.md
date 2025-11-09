# New HDF5 File Creation Feature

## Summary

Added a "New HDF5 File…" feature to the vibehdf5 GUI that allows users to create new empty HDF5 files from scratch directly in the application.

## Changes Made

### 1. GUI Action (`hdf5_viewer.py`)

**Added in `_create_actions()` method:**
```python
self.act_new = QAction("New HDF5 File…", self)
self.act_new.setShortcut("Ctrl+N")
self.act_new.triggered.connect(self.new_file_dialog)
```

**Added to toolbar in `_create_toolbar()` method:**
```python
tb.addAction(self.act_new)  # Added as first action in toolbar
```

### 2. New Method: `new_file_dialog()`

**Location:** `vibehdf5/hdf5_viewer.py`

**Functionality:**
- Opens a "Save File" dialog for the user to specify the new file location and name
- Automatically adds `.h5` extension if not provided
- Checks if file already exists and prompts for overwrite confirmation
- Creates an empty HDF5 file using `h5py.File(filepath, "w")`
- Loads the newly created file in the viewer
- Shows success message in status bar
- Handles errors gracefully with user-friendly dialogs

**Code:**
```python
def new_file_dialog(self) -> None:
    """Create a new HDF5 file."""
    last_dir = os.getcwd()
    filepath, _ = QFileDialog.getSaveFileName(
        self,
        "Create New HDF5 File",
        last_dir,
        "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
    )
    if not filepath:
        return

    # Add .h5 extension if no extension provided
    if not filepath.endswith(('.h5', '.hdf5')):
        filepath += '.h5'

    # Check if file already exists
    if os.path.exists(filepath):
        resp = QMessageBox.question(
            self,
            "File exists",
            f"File '{filepath}' already exists. Overwrite?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

    try:
        import h5py
        # Create a new empty HDF5 file
        with h5py.File(filepath, "w"):
            pass

        # Load the newly created file
        self.load_hdf5(filepath)
        self.statusBar().showMessage(f"Created new HDF5 file: {filepath}", 5000)
    except Exception as exc:
        QMessageBox.critical(
            self,
            "Failed to create file",
            f"Could not create HDF5 file:\n{filepath}\n\n{exc}",
        )
```

### 3. Documentation Updates

**README.md:**
- Added "Creating New Files" section with step-by-step instructions
- Updated keyboard shortcuts list to include `Ctrl+N`
- Documented the workflow for creating and populating new files

## User Workflow

1. **Create New File:**
   - Click "New HDF5 File…" button in toolbar or press `Ctrl+N`
   - Choose location and filename in dialog
   - File is created and automatically loaded

2. **Add Content:**
   - Use "Add Files…" (`Ctrl+Shift+F`) to add individual files
   - Use "Add Folder…" (`Ctrl+Shift+D`) to import directory structures
   - Drag & drop files/folders directly from file manager

3. **Manage Content:**
   - View datasets and attributes in the tree
   - Preview data in the side panel
   - Delete items via right-click context menu
   - Export by dragging items out of the tree

## Technical Details

- **File Format:** Creates standard HDF5 files compatible with h5py and other HDF5 tools
- **Initial State:** New files contain only the root group (empty)
- **File Extension:** Automatically appends `.h5` if user doesn't specify `.h5` or `.hdf5`
- **Overwrite Protection:** Prompts user before overwriting existing files
- **Error Handling:** Shows clear error dialogs if file creation fails (permissions, disk space, etc.)

## Testing

A test script has been created at `test_new_file_feature.py` that demonstrates:
- Creating a new empty HDF5 file
- Verifying the file can be opened and read
- Adding content (groups, datasets, attributes)
- Verifying the final structure

Run the test with:
```bash
python test_new_file_feature.py
```

## Benefits

1. **Complete Workflow:** Users can now create, populate, and manage HDF5 files entirely within the GUI
2. **No Command Line Required:** No need to use external tools or scripts to create files
3. **Immediate Productivity:** New files are ready to receive content immediately after creation
4. **Consistent Experience:** Same UI patterns for creating and opening files
5. **Safe Operations:** Built-in overwrite protection and error handling

## Related Features

This feature complements existing functionality:
- **Add Files/Folders:** Immediately usable on new files
- **Drag & Drop:** Works seamlessly with newly created files
- **Delete Operations:** Can be used to clean up content in new files
- **Export:** Created files can be managed like any other HDF5 file

## Future Enhancements

Potential improvements for future versions:
- Template support (create new files from predefined structures)
- Metadata initialization (add standard attributes on creation)
- Batch file creation
- Integration with project templates
