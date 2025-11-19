# vibehdf5 Documentation

This directory contains the Sphinx documentation for vibehdf5.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install directly from the requirements file:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

From the `docs/` directory:

```bash
make html
```

The generated HTML documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

### Build PDF Documentation (requires LaTeX)

```bash
make latexpdf
```

### Clean Build Files

```bash
make clean
```

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index page
- `installation.rst` - Installation instructions
- `usage.rst` - User guide
- `api/` - API reference documentation
  - `modules.rst` - API modules index
  - `hdf5_viewer.rst` - Main viewer module documentation
  - `hdf5_tree_model.rst` - Tree model documentation
  - `syntax_highlighter.rst` - Syntax highlighter documentation
  - `utilities.rst` - Utilities module documentation

## Documentation Style

The documentation uses:
- **reStructuredText** (.rst) format
- **Sphinx RTD Theme** for HTML output
- **Napoleon** extension for Google and NumPy style docstrings
- **Autodoc** for automatic API documentation from docstrings
- **Intersphinx** for linking to Python, NumPy, h5py, and pandas documentation

## Writing Docstrings

All public functions, classes, and methods should have docstrings following the Google or NumPy style:

```python
def example_function(param1, param2):
    """Brief description of function.

    More detailed description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
    pass
```

## Online Documentation

The documentation can be hosted on:
- [Read the Docs](https://readthedocs.org/)
- GitHub Pages
- Any static hosting service

For Read the Docs, simply connect your GitHub repository and the documentation will be built automatically from the `docs/` directory.
