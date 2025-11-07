# HDF5 Tree Viewer (PySide6)

A lightweight GUI to browse the structure of an HDF5 file using a tree view. It shows groups, datasets (with shape and dtype), and attributes.

## How to run

On macOS with pixi (environment provided in `env/pixi.toml`):

```bash
# Open a shell with the project environment
pixi shell

# From the project root, launch the viewer
python hdf5_viewer.py test_files.h5
```

Alternatively, run without an argument and you can choose a file from the dialog:

```bash
python hdf5_viewer.py
```

## Files

- `hdf5_viewer.py` — GUI application entrypoint with `QTreeView` and toolbar actions.
- `hdf5_tree_model.py` — Reusable Qt model (QStandardItemModel) that builds the HDF5 tree.
- `hdf5_utilities.py` — Utilities to archive and inspect directory structures in HDF5 (existing).
- `test_files.h5` — Sample HDF5 file in the repo root that will auto-load on startup if present.

## Notes

- Double-click or use the toolbar to expand/collapse the tree.
- Attributes are grouped under an "Attributes" node for each group.
- This viewer loads the entire tree on open (simple and effective for small/medium files). For very large files, consider lazy loading.

## Troubleshooting

- If the app doesn't start and you're using a Mac with Apple Silicon, make sure you're inside the pixi environment so that the correct PySide6 build is used.
- If you see import/type warnings in editors (about PySide6 stubs), they are harmless at runtime.