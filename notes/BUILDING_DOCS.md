# Building vibehdf5 Documentation

Quick reference for building the Sphinx documentation.

## Using pixi environment

```bash
# Enter pixi environment
cd env
pixi shell

# Navigate to docs directory
cd ../docs

# Build HTML documentation
make html

# View the documentation
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
# or just open _build/html/index.html in your browser
```

## Alternative build commands

```bash
# Build PDF (requires LaTeX)
make latexpdf

# Build EPUB
make epub

# Clean build files
make clean

# Rebuild everything
make clean html
```

## Viewing documentation locally

After building, the HTML documentation is located at:
```
docs/_build/html/index.html
```

## Deploying to Read the Docs

1. Create account at https://readthedocs.org/
2. Connect your GitHub repository
3. The `.readthedocs.yaml` configuration file is already set up
4. Documentation will build automatically on each commit

## Documentation structure

- Main page: `docs/index.rst`
- Installation guide: `docs/installation.rst`
- Usage guide: `docs/usage.rst`
- API reference: `docs/api/` directory
- Configuration: `docs/conf.py`

All API documentation is auto-generated from your docstrings using Sphinx autodoc.
