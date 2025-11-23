#!/usr/bin/env bash
# Build vibehdf5 as a standalone executable using Nuitka
# Nuitka compiles Python to C and produces faster executables than PyInstaller

set -e

echo "==================================="
echo "Building vibehdf5 with Nuitka..."
echo "==================================="

# Check if running in pixi environment
if ! command -v nuitka &> /dev/null; then
    echo "ERROR: Nuitka not found!"
    echo "Install it with: pixi add nuitka --manifest-path ./env/pixi.toml"
    echo "Or run: pixi shell --manifest-path ./env/pixi.toml"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Using Python $PYTHON_VERSION"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf vibehdf5.dist vibehdf5.build vibehdf5.onefile-build

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
    OUTPUT_NAME="vibehdf5"
    EXTRA_ARGS="--macos-create-app-bundle --macos-app-icon=media/icon.icns"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
    OUTPUT_NAME="vibehdf5"
    EXTRA_ARGS=""
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    PLATFORM="Windows"
    OUTPUT_NAME="vibehdf5.exe"
    EXTRA_ARGS="--windows-icon-from-ico=icon.ico"
else
    PLATFORM="Unknown"
    OUTPUT_NAME="vibehdf5"
    EXTRA_ARGS=""
fi

echo "Building for: $PLATFORM"

# Run Nuitka with optimized settings
python -m nuitka \
    --standalone \
    --onefile \
    --enable-plugin=pyside6 \
    --include-package=vibehdf5 \
    --include-package=h5py \
    --include-package=numpy \
    --include-package=pandas \
    --include-package=matplotlib \
    --noinclude-pytest-mode=nofollow \
    --noinclude-IPython-mode=nofollow \
    --noinclude-unittest-mode=nofollow \
    --disable-console \
    --output-filename="$OUTPUT_NAME" \
    --company-name="Jacob Williams" \
    --product-name="vibehdf5" \
    --file-version="1.1.0" \
    --product-version="1.1.0" \
    --file-description="HDF5 File Viewer & Manager" \
    --copyright="Copyright (c) 2024 Jacob Williams" \
    $EXTRA_ARGS \
    run_vibehdf5.py

echo ""
echo "==================================="
echo "Build complete!"
echo "==================================="

if [[ "$PLATFORM" == "macOS" ]]; then
    echo "Application: vibehdf5.app"
    echo "Launch with: open vibehdf5.app"
    echo ""
    echo "To distribute:"
    echo "  Copy vibehdf5.app to /Applications/"
    echo "  Or compress for sharing: ditto -c -k --sequesterRsrc --keepParent vibehdf5.app vibehdf5.app.zip"
elif [[ "$PLATFORM" == "Linux" ]]; then
    echo "Executable: vibehdf5.dist/$OUTPUT_NAME"
    echo "Launch with: ./vibehdf5.dist/$OUTPUT_NAME"
    echo ""
    echo "To distribute:"
    echo "  Copy to /usr/local/bin: sudo cp vibehdf5.dist/$OUTPUT_NAME /usr/local/bin/"
    echo "  Or compress: tar -czf vibehdf5-linux.tar.gz vibehdf5.dist/"
elif [[ "$PLATFORM" == "Windows" ]]; then
    echo "Executable: vibehdf5.dist\\$OUTPUT_NAME"
    echo "Launch with: vibehdf5.dist\\$OUTPUT_NAME"
    echo ""
    echo "To distribute:"
    echo "  Compress vibehdf5.dist folder"
    echo "  Or create installer with Inno Setup"
fi

echo ""
echo "Note: First launch may be slower as Nuitka initializes cached resources"
