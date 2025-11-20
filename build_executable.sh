#!/bin/bash
# Build script for creating vibehdf5 standalone executable

set -e  # Exit on error

echo "=========================================="
echo "Building vibehdf5 standalone executable"
echo "=========================================="
echo ""

# Check if pyinstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.spec.bak

# Run PyInstaller
echo ""
echo "Running PyInstaller..."
pyinstaller vibehdf5.spec

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Build completed successfully!"
    echo "=========================================="
    echo ""

    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS application created at:"
        echo "  dist/vibehdf5.app"
        echo ""
        echo "To run the application:"
        echo "  open dist/vibehdf5.app"
        echo ""
        echo "To install to Applications folder:"
        echo "  cp -r dist/vibehdf5.app /Applications/"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Linux executable created at:"
        echo "  dist/vibehdf5"
        echo ""
        echo "To run the executable:"
        echo "  ./dist/vibehdf5"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "Windows executable created at:"
        echo "  dist/vibehdf5.exe"
        echo ""
        echo "To run the executable:"
        echo "  dist\\vibehdf5.exe"
    fi
else
    echo ""
    echo "=========================================="
    echo "Build failed!"
    echo "=========================================="
    exit 1
fi
