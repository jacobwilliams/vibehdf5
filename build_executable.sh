#!/bin/bash
# Build script for creating vibehdf5 standalone executable

set -e  # Exit on error

echo "=========================================="
echo "Building vibehdf5 standalone executable"
echo "=========================================="
echo ""

pixi run --manifest-path ./env/pixi.toml pyinstaller --clean vibehdf5.spec --noconfirm